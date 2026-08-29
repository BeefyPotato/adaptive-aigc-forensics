# Domain docs

This repository uses a single domain context.

Before exploring the codebase, read:

- `CONTEXT.md` at the repository root, when present
- Relevant architectural decision records under `docs/adr/`

If these files do not exist, proceed silently. `/domain-modeling`, normally
reached through `/grill-with-docs`, creates them when terminology or decisions
are resolved.

Use the vocabulary defined in `CONTEXT.md` consistently in issues, tests,
implementation, and documentation. If a necessary concept is absent, note it
for domain modelling instead of silently introducing conflicting terminology.

If proposed work conflicts with an existing ADR, identify the conflict
explicitly rather than silently overriding the decision.
