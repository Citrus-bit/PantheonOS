---
id: paper_writing_evidence
name: Paper Writing Evidence Layer
description: Evidence layer for paper search, OA paper fetch, claim-evidence maps, citation grounding, evidence summaries, reranking, and context-bound answers.
tags: [paper_writing, evidence, citations, rag]
---

# Evidence Skills

Evidence skills decide where facts come from. Writers may not invent facts or
use model memory as support for manuscript claims.

| Need | File |
|---|---|
| Search candidate papers | [paper_search.md](./paper_search.md) |
| Fetch an open-access PDF by identifier | [paper_fetch.md](./paper_fetch.md) |
| Register claims and support | [evidence_registry.md](./evidence_registry.md) |
| Ground citations for text segments | [citation_grounding.md](./citation_grounding.md) |
| Summarize retrieved evidence | [evidence_summary.md](./evidence_summary.md) |
| Rerank and attribute sentences | [rerank_and_attribution.md](./rerank_and_attribution.md) |
| Answer only from provided context | [context_answering.md](./context_answering.md) |
| Write data/code availability | [data_availability.md](./data_availability.md) |

Default order: search/fetch -> summarize -> register evidence -> draft -> check.
