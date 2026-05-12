---
id: paper_writing_paper_fetch
name: Paper Fetch
description: Legal open-access paper PDF fetching protocol using DOI/title/PMID/arXiv identifiers with JSON envelopes and idempotency.
tags: [paper_writing, paper_fetch, open_access]
---

# Paper Fetch

Use when the user provides DOI, PMID, arXiv ID, title, or asks to retrieve a
paper PDF for evidence review.

## Allowed Routes

Only use legal open paths:

- Unpaywall
- Semantic Scholar
- arXiv
- PubMed Central
- bioRxiv / medRxiv
- publisher direct OA link
- user-provided licensed/local PDF

Do not use Sci-Hub or any access-control bypass.

## Output Envelope

Return or record:

```json
{
  "ok": true,
  "input": {"doi": "...", "title": "..."},
  "source": "unpaywall|semantic_scholar|arxiv|pmc|biorxiv|publisher|user_file",
  "pdf_path": "materials/papers/<slug>.pdf",
  "metadata": {"title": "...", "doi": "...", "year": 2026},
  "idempotency_key": "doi-or-title-hash"
}
```

Failure envelope:

```json
{
  "ok": false,
  "input": {"title": "..."},
  "error_type": "not_found|not_oa|ambiguous|network|blocked",
  "retryable": false,
  "next_action": "ask_user_for_pdf_or_identifier"
}
```

Sources: Agents365-ai/paper-fetch/skills/paper-fetch/SKILL.md,
K-Dense citation-management/SKILL.md.
