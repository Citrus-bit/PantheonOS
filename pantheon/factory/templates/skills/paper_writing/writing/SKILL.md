---
id: paper_writing_writer
name: Evidence-Bound Paper Writer
description: Section-level writing skill for evidence-bound academic prose, IMRAD papers, grants, reports, talks, and response letters.
tags: [paper_writing, writing, imrad]
---

# Evidence-Bound Paper Writer

Use only after triage, outline, and evidence boundaries exist.

## Drafting Rules

- Write `draft/paper.md` as the content source of truth.
- Keep each core claim within `claim_evidence_map.md`.
- Use `missing` evidence to ask for material, downgrade, or remove claims.
- Do not invent data, citations, mechanisms, reviewer changes, or availability
  statements.
- For major rewrites, shape from raw material to candidate openings to
  paragraph-by-paragraph structure before polishing.

## Section Roles

| Section | Role | Quality gate |
|---|---|---|
| Title | searchable, precise, not inflated | no vague clever title |
| Abstract | problem, gap, method, key result, meaning | no result-free impact claim |
| Introduction | known -> gap -> insufficiency -> approach -> contribution | gap and contribution align |
| Related work | themed positioning | not chronological paper dump |
| Methods | reproducible protocol | data/software/parameters/statistics clear |
| Results | question -> method -> observation -> quantitative result -> interpretation | each result points to figure/table/data |
| Discussion | finding, relation to literature, mechanism, limits, future | no new data |
| Limitations | honest boundary and risk | no hidden fatal flaw |
| Conclusion | contribution and boundary | not abstract repetition |

Sources: ResearAI writer.md, Research-Paper-Writing-Skills/SKILL.md,
K-Dense scientific-writing/SKILL.md, mattpocock writing-shape/SKILL.md.
