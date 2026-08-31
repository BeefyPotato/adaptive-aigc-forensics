# Signal Hackathon Configuration Implementation Plan

**Goal:** Complete Issue #6 within the hackathon using the official `hackathon-v1` configuration: 8,064 balanced expert-training draws and the complete 20-condition matrix for 400 deterministic internal-validation sources.

**Architecture:** This is the same Issue-6 signal-only system. The named profile changes only deterministic observation selection. It retains the finalized Issue-3 manifest, source partitions, corruption recipes, balanced sampler, 26-feature representation, MLP, normalization boundaries, checkpoint rule, canonical evaluator, and leakage controls. Profile and acceptance scope travel through the plan, normalization, checkpoint, logits, and final run marker. The unchanged seeded-noise kernel is executed across bounded CPU workers only after byte-identity tests pass.

**Tech stack:** Node.js 22+ ESM, Python 3.12, NumPy, Pillow, Sharp/libvips, `node:test`, and `unittest`.

**Spec:** GitHub Issues #3 and #6, `docs/track5-manifest.md`, and `docs/signal-expert.md`.

## Invariants

- Use the finalized Issue-3 manifest, images, source partitions, corruption harness, and balanced sampler.
- Corrupt decoded native-resolution RGB before shared geometry, luminance, or feature extraction.
- Preserve the exact 16 Fourier + 6 neighbour + 4 residual feature order.
- Fit normalization and signal weights only on `expert-training`; use only `internal-validation` for checkpoint selection and development metrics.
- Never use `fusion-training`, sealed-test labels, or organizer demonstration data in Issue #6.
- Bind `hackathon-v1` to 8,064 training draws, 400 validation sources, and `issue-6-timeboxed-acceptance`; reject mismatches and relabelling.
- Keep generated images, caches, checkpoints, and machine paths under ignored artifact roots.

## Work items

- [x] Add deterministic class-balanced whole-source validation selection and named, fail-closed experiment profiles.
- [x] Carry profile/scope through Python normalization, checkpoint, logits, metrics, and completion provenance.
- [x] Add byte-identical bounded worker parallelism for seeded native-resolution RGB noise.
- [x] Verify the real plan contains 8,064 weighted training draws over all 8,000 training sources and 8,000 validation observations over 400 class-balanced sources.
- [x] Document the time-boxed profile and the optional post-hackathon `issue-6-full-v1` replication.
- [x] Run all Node and Python verification from pinned clean environments.
- [x] Complete the real `hackathon-v1` experiment and validate every published artifact.
- [x] Freeze the selected signal checkpoint, update Issues #6-#11 with exact results and provenance, and independently review the diff.
- [x] Commit and push the feature branch; merge and close #6 only if the production completion marker and all acceptance checks pass.
