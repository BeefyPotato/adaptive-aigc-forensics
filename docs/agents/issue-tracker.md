# Issue tracker: GitHub

Issues and specifications for this repository live in GitHub Issues. Use the
`gh` CLI for all operations and infer the repository from `git remote -v`.

## Conventions

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --comments`
- List: `gh issue list --state open --json number,title,body,labels,comments`
- Comment: `gh issue comment <number> --body "..."`
- Add or remove labels with `gh issue edit`
- Close: `gh issue close <number> --comment "..."`

Pull requests are not treated as incoming triage requests.

When a skill says "publish to the issue tracker," create a GitHub issue.
When a skill says "fetch the relevant ticket," use `gh issue view`.

## Wayfinding

A Wayfinder map is one issue labelled `wayfinder:map`. Decision tickets are
child issues labelled by type: `wayfinder:research`, `wayfinder:prototype`,
`wayfinder:grilling`, or `wayfinder:task`.

Use GitHub sub-issues and native issue dependencies when available. Otherwise,
link children through the map's task list and record blockers as
`Blocked by: #<number>`.

An unblocked, unassigned child is eligible to be claimed. Claim it by assigning
the current GitHub user, resolve it with a comment, close it, and add the
resulting decision link to the map.
