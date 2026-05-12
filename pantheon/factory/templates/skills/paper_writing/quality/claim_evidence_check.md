---
id: paper_writing_claim_evidence_check
name: Claim Evidence Check
description: Audit manuscript claims against claim_evidence_map.md and flag unsupported, overbroad, or missing evidence.
tags: [paper_writing, evidence, quality]
---

# Claim Evidence Check

Use before any final output.

## Output

Write `quality/claim_evidence_report.md`:

| Claim ID | Draft location | Evidence | Status | Risk | Required action |
|---|---|---|---|---|---|

Statuses: `supported`, `partial`, `unsupported`, `missing`, `overbroad`,
`conflicting`.

Rules:

- Unsupported claims must be removed, downgraded, or marked for user input.
- Partial evidence requires narrower wording.
- Claims about novelty, mechanism, generalization, or significance receive extra
  scrutiny.

Sources: paper-review.md, DeepScientist review/SKILL.md.
