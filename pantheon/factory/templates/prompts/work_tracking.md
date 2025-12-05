---
id: work_tracking
name: Work Tracking
description: File-based planning and todo management guidance
---

## Work Tracking with Files

Use file-based tracking for task management and planning. This provides transparency and persistence.

**File Structure:**
- `PLAN.md` - High-level planning and approach documentation
- `TODO.md` - Task list with status tracking

**TODO.md Format:**
```markdown
# Task List

## In Progress
- [ ] Current task description

## Pending
- [ ] Next task description
- [ ] Another pending task

## Completed
- [x] Finished task description
```

**PLAN.md Format:**
```markdown
# Plan: [Goal/Feature Name]

## Overview
Brief description of what we're building/solving

## Approach
1. Step one
2. Step two
3. Step three

## Considerations
- Important notes
- Risks or dependencies

## Progress
- Current status and findings
```

**When to Use:**
- **TODO.md**: For tracking concrete tasks, multi-step work, or when you need to show progress
- **PLAN.md**: For complex tasks requiring research, design decisions, or when approach is unclear

**Workflow:**
1. Create/update TODO.md or PLAN.md at the start of non-trivial work
2. Mark tasks as in-progress when starting (`- [ ]` under "In Progress")
3. Update status as you complete work (`- [x]` and move to "Completed")
4. Add new discovered tasks to "Pending"
5. Keep files updated to reflect current state

**Best Practices:**
- Keep task descriptions concise and actionable
- Update files immediately after completing steps
- Remove or archive completed work periodically
- Use PLAN.md to document decisions and rationale
