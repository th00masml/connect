"""Model adapters.

Every model implements  generate(prompts, constrained=False) -> list[Generation].
Constrained decoding means: for choice prompts, restrict the answer to the allowed
tokens (HF: by scoring each candidate continuation; API: by post-hoc mapping, flagged
as soft); for free-form prompts with a regex, use a grammar-constrained decoder when one
is available.

Dummy models make the diagnostics testable without a GPU and encode the failure modes
the instrument is meant to detect.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Protocol

from .data import Item
from .render import Prompt
from .retrieval import nearest_neighbor


@dataclass
class Generation:
    text: str
    confidence: float | None = None
    meta: dict = field(default_factory=dict)


class Model(Protocol):
    name: str

    def generate(self, prompts: list[Prompt], constrained: bool = False) -> list[Generation]: ...


def _token_for(prompt: Prompt, option_text: str) -> str | None:
    if prompt.choice_map:
        for tok, opt in prompt.choice_map.items():
            if opt == option_text:
                return tok
    return None


# --- dummy models ------------------------------------------------------------------


class OracleModel:
    """Answers correctly with probability `accuracy`, else a uniformly wrong option.
    Ignores surface form entirely, so format sensitivity is zero by construction."""

    def __init__(self, gold: dict[str, str], accuracy: float = 1.0, seed: int = 0, name: str | None = None):
        self.gold, self.accuracy, self.seed = gold, accuracy, seed
        self.name = name or f"oracle@{accuracy:.2f}"

    def generate(self, prompts: list[Prompt], constrained: bool = False) -> list[Generation]:
        out = []
        for p in prompts:
            rng = random.Random(f"{self.seed}:{p.item_id}")
            gold = self.gold[p.item_id]
            right = rng.random() < self.accuracy
            if p.choice_map:
                tok = _token_for(p, gold)
                if right and tok is not None:
                    out.append(Generation(tok, 0.9))
                else:
                    others = [t for t, o in p.choice_map.items() if o != gold] or list(p.choice_map)
                    out.append(Generation(rng.choice(others), 0.6))
            else:
                out.append(Generation(gold if right else "no idea", 0.9 if right else 0.3))
        return out


class FirstChoiceModel:
    """Always picks the first listed option: pure position bias."""

    name = "first-choice"

    def generate(self, prompts: list[Prompt], constrained: bool = False) -> list[Generation]:
        return [Generation(p.choices[0] if p.choices else "", 0.5) for p in prompts]


class ConstantModel:
    """Always answers the same option text (label-prior collapse)."""

    def __init__(self, answer: str):
        self.answer = answer
        self.name = f"constant:{answer}"

    def generate(self, prompts: list[Prompt], constrained: bool = False) -> list[Generation]:
        out = []
        for p in prompts:
            tok = _token_for(p, self.answer)
            out.append(Generation(tok if tok is not None else self.answer, 0.99))
        return out


class RetrievalModel:
    """Nearest-neighbour lookup over training items: the retrieval shortcut made explicit."""

    def __init__(self, train: list[Item], method: str = "fuzzy", name: str | None = None):
        self.texts = [i.text() for i in train]
        self.labels = [i.label for i in train]
        self.method = method
        self.name = name or f"retrieval:{method}"

    def generate(self, prompts: list[Prompt], constrained: bool = False) -> list[Generation]:
        preds, sims = nearest_neighbor(self.texts, self.labels, [p.text for p in prompts], self.method)
        out = []
        for p, pred, sim in zip(prompts, preds, sims):
            tok = _token_for(p, pred)
            out.append(Generation(tok if tok is not None else pred, float(sim)))
        return out


class NoisyFormatModel:
    """Knows the answer with probability `accuracy`, but when unconstrained wraps it in chatter
    that defeats strict decoding with probability `noise`. Constrained decoding removes the
    chatter and changes nothing else. This is the signature of a positive semantic-validity gap."""

    def __init__(self, gold: dict[str, str], accuracy: float = 0.6, noise: float = 0.5, seed: int = 0):
        self.gold, self.accuracy, self.noise, self.seed = gold, accuracy, noise, seed
        self.name = f"noisy-format@{accuracy:.2f}/{noise:.2f}"

    def generate(self, prompts: list[Prompt], constrained: bool = False) -> list[Generation]:
        out = []
        for p in prompts:
            rng = random.Random(f"{self.seed}:{p.item_id}")
            gold = self.gold[p.item_id]
            right = rng.random() < self.accuracy
            noisy = rng.random() < self.noise
            if p.choice_map:
                tok = _token_for(p, gold)
                wrong = [t for t, o in p.choice_map.items() if o != gold] or list(p.choice_map)
                ans_tok = tok if (right and tok) else rng.choice(wrong)
                ans_text = p.choice_map[ans_tok]
                if constrained or not noisy:
                    out.append(Generation(ans_tok, 0.8))
                else:
                    out.append(Generation(f"Let me think about this carefully. I believe the answer is {ans_text}, "
                                          f"though option {rng.choice(list(p.choice_map))} was tempting.", 0.8))
            else:
                ans = gold if right else "~" + gold
                if constrained or not noisy:
                    out.append(Generation(ans, 0.8))
                else:
                    out.append(Generation(f"Sure! Here is a formula that works: {ans} . Hope that helps.", 0.8))
        return out


# --- real adapters (lazy imports) -------------------------------------------------


class HFModel:
    """transformers causal LM. Greedy decoding; choice constraint by candidate log-prob scoring;
    regex constraint via `outlines` if installed, else falls back to unconstrained and flags it."""

    def __init__(self, model_id: str, device: str | None = None, max_new_tokens: int = 48, dtype: str | None = None):
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.torch = torch
        self.name = f"hf:{model_id}"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {}
        if dtype:
            kw["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def _wrap(self, text: str) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        return text + "\n"

    def _score_choices(self, text: str, choices: list[str]) -> list[float]:
        torch = self.torch
        base = self.tok(self._wrap(text), return_tensors="pt").input_ids.to(self.device)
        scores = []
        with torch.no_grad():
            for c in choices:
                cont = self.tok(c, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                ids = torch.cat([base, cont], dim=1)
                logits = self.model(ids).logits[0, base.shape[1] - 1:-1]
                lp = torch.log_softmax(logits.float(), dim=-1).gather(1, cont[0].unsqueeze(1)).sum().item()
                scores.append(lp)
        return scores

    def generate(self, prompts: list[Prompt], constrained: bool = False) -> list[Generation]:
        torch = self.torch
        out = []
        for p in prompts:
            if constrained and p.choices:
                lps = self._score_choices(p.text, p.choices)
                m = max(lps)
                probs = [math_exp(x - m) for x in lps]
                z = sum(probs)
                j = max(range(len(lps)), key=lps.__getitem__)
                out.append(Generation(p.choices[j], probs[j] / z, {"constraint": "choices"}))
                continue
            if constrained and p.regex:
                try:
                    import outlines  # type: ignore

                    gen = outlines.generate.regex(outlines.models.Transformers(self.model, self.tok), p.regex)
                    out.append(Generation(gen(self._wrap(p.text), max_tokens=self.max_new_tokens), None, {"constraint": "regex"}))
                    continue
                except Exception as e:  # pragma: no cover
                    meta = {"constraint": "regex-unavailable", "error": str(e)[:200]}
            else:
                meta = {"constraint": "none"}
            enc = self.tok(self._wrap(p.text), return_tensors="pt").to(self.device)
            with torch.no_grad():
                res = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          output_scores=True, return_dict_in_generate=True,
                                          pad_token_id=self.tok.pad_token_id)
            seq = res.sequences[0, enc.input_ids.shape[1]:]
            lps = []
            for step, tid in zip(res.scores, seq):
                lps.append(torch.log_softmax(step[0].float(), dim=-1)[tid].item())
            conf = math_exp(sum(lps) / len(lps)) if lps else None
            out.append(Generation(self.tok.decode(seq, skip_special_tokens=True), conf, meta))
        return out


def math_exp(x: float) -> float:
    import math

    return math.exp(x)


class OpenAICompatModel:
    """Chat-completions model behind an OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, ...).
    Choice constraint is soft: the free output is mapped to the nearest allowed token."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 64, system: str | None = None):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(base_url=base_url, api_key=api_key) if (base_url or api_key) else OpenAI()
        self.model, self.max_tokens, self.system = model, max_tokens, system
        self.name = f"openai:{model}"

    def generate(self, prompts: list[Prompt], constrained: bool = False) -> list[Generation]:
        out = []
        for p in prompts:
            text = p.text
            if constrained and p.choices:
                text += "\n\nRespond with exactly one of: " + ", ".join(p.choices) + "."
            msgs = ([{"role": "system", "content": self.system}] if self.system else []) + [{"role": "user", "content": text}]
            kw = dict(model=self.model, messages=msgs, temperature=0, max_tokens=self.max_tokens)
            try:
                res = self.client.chat.completions.create(logprobs=True, **kw)
            except Exception:
                res = self.client.chat.completions.create(**kw)
            ch = res.choices[0]
            raw = ch.message.content or ""
            conf = None
            try:
                lps = [t.logprob for t in ch.logprobs.content]
                conf = math_exp(sum(lps) / len(lps)) if lps else None
            except Exception:
                pass
            meta = {"constraint": "none"}
            if constrained and p.choices:
                tok = p.choice_map and _token_for(p, p.decode(raw))
                if not tok:
                    m = re.search("|".join(re.escape(c) for c in sorted(p.choices, key=len, reverse=True)), raw)
                    tok = m.group(0) if m else p.choices[0]
                raw, meta = tok, {"constraint": "soft-choices"}
            out.append(Generation(raw, conf, meta))
        return out


def from_spec(spec: str, *, gold: dict[str, str] | None = None, train: list[Item] | None = None) -> Model:
    """Parse a CLI model spec.
    dummy:oracle:0.8 | dummy:first | dummy:constant:TRUE | dummy:retrieval[:method] | dummy:noisy:0.6:0.5
    hf:<model_id>  |  openai:<model>[@base_url]
    """
    parts = spec.split(":")
    if parts[0] == "dummy":
        kind = parts[1]
        if kind == "oracle":
            return OracleModel(gold or {}, float(parts[2]) if len(parts) > 2 else 1.0)
        if kind == "first":
            return FirstChoiceModel()
        if kind == "constant":
            return ConstantModel(parts[2])
        if kind == "retrieval":
            return RetrievalModel(train or [], parts[2] if len(parts) > 2 else "fuzzy")
        if kind == "noisy":
            return NoisyFormatModel(gold or {}, float(parts[2]) if len(parts) > 2 else 0.6,
                                    float(parts[3]) if len(parts) > 3 else 0.5)
        raise ValueError(f"unknown dummy model {kind}")
    if parts[0] == "hf":
        return HFModel(":".join(parts[1:]))
    if parts[0] == "openai":
        rest = ":".join(parts[1:])
        model, _, base = rest.partition("@")
        return OpenAICompatModel(model, base_url=base or None)
    raise ValueError(f"unknown model spec {spec}")
