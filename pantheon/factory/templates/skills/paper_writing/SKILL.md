---
id: paper_writing_skills_index
name: Paper Writing Skills Index
description: |
  Routing and workflow skills for the Paper Write Team. Use when users ask for
  manuscript drafting, journal or conference papers, grant proposals, lab
  reports, group-meeting reports, talks, workshop notes, reviewer rebuttals,
  academic HTML/PDF/LaTeX output, citation grounding, evidence checking, or
  paper-writing quality review.
tags: [paper_writing, manuscript, grant, rebuttal, citation, html, latex]
---

# Paper Writing Skills

This skill is the only paper-writing entry point. Keep `paper_write_team` as the
team interface; route inside this skill family instead of adding new teams.

## Routing + Sequential Pipeline

Always run these phases in order. Load only the referenced files needed for the
current scenario.

| Phase | Entry criteria | Actions | Exit criteria |
|---|---|---|---|
| 0. Triage | User request and any UI scenario labels | Read [workflow/triage.md](./workflow/triage.md), choose `scenario_id`, `format_id`, `theme_id`, language, audience, outputs, constraints | `triage.md` exists or is updated |
| 1. Materials and evidence | Triage is known | Read relevant scenario skill, inventory materials, fetch/search papers only when needed, build evidence registry | `materials/inventory.md` and/or `claim_evidence_map.md` |
| 2. Outline and claim boundary | Evidence and materials are known | Read [workflow/paper_outline.md](./workflow/paper_outline.md), add figure storyline or knowledge lineage when needed | `paper_view` and `evidence_view` outline |
| 3. Section drafting | Outline and evidence boundary exist | Read [writing/SKILL.md](./writing/SKILL.md), draft Markdown only within evidence bounds | `draft/paper.md` |
| 4. Quality gates | Draft exists | Read [quality/SKILL.md](./quality/SKILL.md), run scenario-specific gates | quality reports under `quality/` |
| 5. Editable output | Draft and quality notes exist | Read [formats/html_editable_contract.md](./formats/html_editable_contract.md), use the selected template/theme | `report/<slug>_preview.html` and final resume packet |

## Scenario Routing

| User intent | `scenario_id` | Required skill | Format/theme | Quality gates |
|---|---|---|---|---|
| manuscript, paper submission, write a paper | `paper_submission` | [scenarios/paper_submission.md](./scenarios/paper_submission.md) | `conference_paper` or `journal_article`, `editable_article` | claim evidence, reviewer rubric, format lint |
| journal article, SCI, Nature-style paper | `journal_article` | [scenarios/journal_article.md](./scenarios/journal_article.md) | `journal_article`, `journal_article` | data availability, citation, reviewer rubric |
| conference paper, workshop paper, double column | `conference_paper` | [scenarios/conference_paper.md](./scenarios/conference_paper.md) | `conference_paper`, `conference_paper` | page limit, baseline/evaluation, reviewer rubric |
| grant, proposal, funding application | `grant_proposal` | [scenarios/grant_proposal.md](./scenarios/grant_proposal.md) | `grant_application`, `grant_form` | gap-aim-route feasibility, word limits |
| lab report, experiment report | `lab_report` | [scenarios/lab_report.md](./scenarios/lab_report.md) | `lab_report`, `lab_report` | reproducibility, raw observation/result separation |
| group meeting, weekly report | `group_report` | [scenarios/group_report.md](./scenarios/group_report.md) | `group_report`, `group_report` | evidence summary, discussion questions |
| conference talk, workshop sharing | `conference_talk` or `workshop_share` | [scenarios/conference_talk.md](./scenarios/conference_talk.md) or [scenarios/workshop_share.md](./scenarios/workshop_share.md) | `conference_talk`, `group_report` | storyline, speaker notes |
| reviewer comments, rebuttal, revision response | `revision_response` | [scenarios/revision_response.md](./scenarios/revision_response.md) | `revision_response`, `revision_response` | every-comment response, manuscript consistency |

## Required Outputs

For full paper-writing tasks, default to these artifacts unless the user narrows
the request:

- `triage.md`
- `draft/paper.md`
- `report/<slug>_preview.html`
- `quality/claim_evidence_report.md`
- `quality/reviewer_report.md` for submissions, grants, rebuttals, or high-risk tasks

## Template And Protocol Files

| Purpose | File |
|---|---|
| Standard HTML report compatibility | [report_standard.md](./report_standard.md) |
| Academic HTML preview compatibility | [report_academic.md](./report_academic.md) |
| Chinese LaTeX compatibility | [latex_cn.md](./latex_cn.md) |
| English LaTeX compatibility | [latex_en.md](./latex_en.md) |
| Editable HTML block contract | [formats/html_editable_contract.md](./formats/html_editable_contract.md) |
| Scenario format contracts | [formats/scenario_formats.md](./formats/scenario_formats.md) |
| Kami-style academic theme | [themes/kami_academic.md](./themes/kami_academic.md) |

## Non-Negotiable Rules

- Do not write unsupported claims. Mark missing evidence and downgrade or remove
  the claim.
- Do not invent citations, DOI, accession IDs, page numbers, data repositories,
  reviewer changes, or experimental results.
- Do not let a search result become evidence until it is summarized, attributed,
  and bound to a specific claim.
- Do not use Sci-Hub or any access-control bypass. Open PDF fetching may use only
  legal OA routes described in [evidence/paper_fetch.md](./evidence/paper_fetch.md).
- Do not output screenshot-like reports. Main text must remain semantic and
  editable HTML.
- Keep each `SKILL.md` below 500 lines. Put long templates, guidelines, and
  examples in adjacent one-hop files.

## Sources Used To Shape This Skill Family

- Anthropic skill design: <https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md>
- Anthropic doc coauthoring: <https://github.com/anthropics/skills/blob/main/skills/doc-coauthoring/SKILL.md>
- Trail of Bits workflow skill design: <https://github.com/trailofbits/skills/blob/main/plugins/workflow-skill-design/skills/designing-workflow-skills/SKILL.md>
- PaperQA evidence RAG: <https://github.com/Future-House/paper-qa/blob/main/README.md>
- OpenScholar attribution: <https://github.com/akariasai/openscholar/blob/main/src/open_scholar.py>
- Nature-style response/figure/citation/data skills: <https://github.com/Yuan1z0825/nature-skills>
- DeepScientist outline/review/rebuttal/writer skills: <https://github.com/ResearAI/DeepScientist>
