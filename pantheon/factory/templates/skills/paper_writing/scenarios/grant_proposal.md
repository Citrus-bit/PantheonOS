---
id: grant_proposal_scenario
name: Grant Proposal Scenario
description: Funding proposal route for grants, proposals, and project applications.
tags: [paper_writing, grant, proposal]
---

# Grant Proposal Scenario

Use when the user asks for funding applications, project proposals, agency-style
forms, or "国自然" style material.

| Field | Contract |
|---|---|
| Trigger | grant, proposal, funding, 立项依据, 国自然, application |
| Inputs | topic, research basis, team, preliminary data, aims, timeline, budget notes |
| Read next | `workflow/research_question.md`, `workflow/knowledge_lineage.md`, `workflow/figure_storyline.md`, `formats/scenario_formats.md` |
| Outputs | form-like editable HTML, aims, research content, route, feasibility, expected outcomes |
| Gates | gap-aim-route consistency, feasibility, innovation boundary, word limits |
| Forbidden | generic innovation claims, unsupported feasibility, invented budget/team facts |

Proposal logic:

```text
field status -> gap -> scientific question -> aims -> work packages
-> technical route -> feasibility -> risk and alternatives -> outputs
```

Sources: research-grants/SKILL.md, strategist/SKILL.md, local design PDF.
