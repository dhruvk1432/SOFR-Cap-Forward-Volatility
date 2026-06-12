# Knowledge-Base Protocol

This repo can use the local NotebookLM mirror established in the sibling
`Options_Portfolio_Model` project.  The hook is configured in
`.codex/config.toml` and points to the local MCP server and source manifest in
that sibling repo.

The public repo does not include the private note corpus, credentials, or raw
data.  The goal is reproducible research hygiene: each extension should state
which theoretical idea it is testing, which data fields are needed, and how the
strategy would fail.

## Research Questions Used For This Extension

- When does implied rate volatility behave like a compensated variance-risk
  premium rather than an unbiased forecast?
- Which filters are required before a rates-vol signal becomes investable:
  transaction costs, margin/funding, liquidity, regime stability, and drawdown?
- How should overlapping volatility-settlement horizons be evaluated without
  overstating sample size?
- What is the correct next step from a cap-vol panel: stylized volatility carry,
  caplet strip valuation, or execution-aware cap/floor structures?

## Implementation Discipline

- Keep all raw workbooks and downloaded public data under `data/raw/`.
- Do not commit `.env`, API keys, provider tokens, or raw market data.
- Keep strategy functions in `src/` and call them from notebooks.
- Treat notebook outputs as research diagnostics, not as proof of deployability.
