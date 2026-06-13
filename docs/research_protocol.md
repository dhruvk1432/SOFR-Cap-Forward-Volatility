# Research Protocol

This repository is designed as a public research artifact, not as a live
caplet trading system. The implementation keeps raw market workbooks
local, publishes aggregate diagnostics, and makes every paper table and
figure reproducible from scripts.

## Public Data Policy

- Keep raw cap-volatility, SOFR swap, and SOFR reference-rate workbooks
  outside version control.
- Store local workbooks in the repo root or `data/raw/`.
- Commit only code, documentation, generated aggregate tables, generated
  figures, tests, and paper sources.
- Do not commit credentials, private notes, provider tokens, or local
  tool configuration.

## Claim Discipline

- Separate curve-construction evidence from trading evidence.
- Report overlapping-horizon tests with HAC errors and effective
  non-overlapping sample counts.
- Treat the carry screen as an investability test in normal-vol basis
  points, not as caplet mark-to-market P&L.
- Require caplet marks, bid/ask, skew, margin, funding, and executable
  structures before claiming production alpha.
