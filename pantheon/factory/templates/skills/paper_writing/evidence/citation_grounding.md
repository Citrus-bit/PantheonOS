---
id: paper_writing_citation_grounding
name: Citation Grounding
description: Ground manuscript segments to citation candidates with support grades and no invented metadata.
tags: [paper_writing, citation, grounding]
---

# Citation Grounding

Use when drafting related work, introduction, discussion, or any claim needing
published support.

## Output

| Segment/claim | Query | Candidate paper | Support grade | Evidence note | Citation metadata |
|---|---|---|---|---|---|

Support grades:

- `strong`: directly supports the claim.
- `partial`: supports a narrower version.
- `background`: relevant context only.
- `conflicting`: contradicts or complicates the claim.

## Rules

- Do not cite a paper only because it shares keywords.
- Do not invent DOI, PMID, venue, pages, or year.
- Prefer exact claim support over famous or high-citation background papers.

Sources: nature-citation/SKILL.md, PaperQA prompts.py.
