---
id: revision_response_scenario
name: Revision Response Scenario
description: Reviewer rebuttal and revision-response route.
tags: [paper_writing, rebuttal, revision, reviewer_response]
---

# Revision Response Scenario

Use when the user provides reviewer comments, editor letters, rebuttal requests,
or asks for a point-by-point response.

| Field | Contract |
|---|---|
| Trigger | reviewer comments, rebuttal, response letter, revision response, 返修 |
| Inputs | editor letter, reviewer comments, current draft, changed text, new evidence |
| Read next | `workflow/revision_loop.md`, `writing/response_letter.md`, `quality/response_consistency_check.md` |
| Outputs | response letter, revision matrix, text deltas, revised draft, editable HTML |
| Gates | every comment has response and change status; response matches manuscript |
| Forbidden | omitting comments, changing reviewer meaning, claiming changes not made |

Response unit:

```text
Reviewer X, Comment Y
Comment:
Response:
Changes Made:
Status: done | partial | declined-with-reason | needs-user-input
```

Sources: nature-response/SKILL.md, DeepScientist rebuttal/SKILL.md.
