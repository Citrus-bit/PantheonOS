---
id: paper_submission_scenario
name: Paper Submission Scenario
description: General manuscript submission route for paper-writing tasks.
tags: [paper_writing, manuscript, submission]
---

# Paper Submission Scenario

Use when the user asks to write, revise, or package a paper but has not locked a
journal/conference-specific structure.

| Field | Contract |
|---|---|
| Trigger | "paper", "manuscript", "投稿", "论文写作", "write a paper" |
| Inputs | research materials, draft, figures/tables, target venue if known |
| Read next | `workflow/material_inventory.md`, `workflow/literature_review.md`, `workflow/paper_outline.md`, `writing/SKILL.md` |
| Outputs | `triage.md`, `draft/paper.md`, `report/paper_preview.html`, quality reports |
| Gates | `claim_evidence_check`, `reviewer_rubric`, `format_lint`, `html_editability_check` |
| Forbidden | choosing a venue-agnostic structure when venue constraints are provided |

Default path:

```text
triage -> material_inventory -> literature_review -> evidence_registry
-> figure_storyline when figures are central -> paper_outline -> draft
-> claim_evidence_check -> reviewer_rubric -> editable HTML
```

Sources: academic-paper/SKILL.md, research-paper-writing/SKILL.md,
paper-outline/SKILL.md, paper-review.md.
