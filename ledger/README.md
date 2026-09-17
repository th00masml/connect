# Ledgers

Two files, two epistemic statuses. The scorer prints the status on the first line and refuses to
call a retrospective ledger evidence.

- `retrospective_papers12.json`: predictions about papers 1 and 2. Their outcomes were published
  before the ledger existed and the instrument was built by inspecting them. This is the
  development set. Scoring it demonstrates the protocol; it validates nothing.
- `prospective_template.json`: the test-set ledger (FOLIO, LogiQA, SynthModal-Control). It stays
  `status: retrospective` until the freeze procedure in its `meta` is completed. Only a ledger
  whose hash was deposited before any model ran on those benchmarks may be set to
  `status: prospective`, and only that score is the paper's result.
