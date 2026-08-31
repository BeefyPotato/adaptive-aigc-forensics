# Hackathon model-training configuration

The official time-boxed Issue #6 run uses `hackathon-v1`. This changes only how many Issue-3 observations are selected:

- 8,064 balanced `expert-training` draws, covering all 8,000 training sources.
- 400 deterministic `internal-validation` sources, 200 per class.
- All 20 clean/corruption conditions for every selected validation source, producing 8,000 validation observations.

Everything else is unchanged: the Issue-3 manifest and source partitions, native-resolution corruption harness, exact 26 signal features, 26-to-16-to-1 MLP, expert-training-only normalization and weight updates, internal-validation-only checkpoint selection, canonical metrics, and fail-closed provenance. The selected checkpoint is therefore the normal frozen Issue #6 signal checkpoint for the hackathon. The 40,320-training-draw/40,000-validation-observation `issue-6-full-v1` run is an optional post-hackathon replication, not a prerequisite for downstream work.

## Downstream handoff

Issue #7 may use the frozen `hackathon-v1` signal checkpoint together with the frozen RGB checkpoint. Expert weights and their normalizations remain frozen. Calibrators and learned static fusion are fit only on `fusion-training`; Issue #7 must record the signal profile and checkpoint revision in its artifacts.

Issue #8 trains the degradation gate only after Issue #7 has frozen expert checkpoints, calibrators, and the static-fusion reference. It may not update either expert and must preserve the same checkpoint/profile provenance.

Issue #9 opens sealed-test labels only once Issue #8 has frozen the entire selection and threshold policy. The smaller Issue #6 profile does not relax the one-time sealed evaluation rule, and organizer demonstration data remains evaluation-only.

Issues #10 and #11 harden and package the selected frozen bundle. Inference and submission metadata must disclose `experiment_profile=hackathon-v1`, its exact source/observation counts, and the selected signal checkpoint revision.

Downstream tickets remain ordered by their scientific dependencies, but no additional "pilot-only" blocker applies. Once Issue #6 has a validated production completion marker and frozen checkpoint, Issue #7 can proceed normally.

## Completion status

The `hackathon-v1` production run completed and passed a full source-pin/cache revalidation on 2026-08-31. Its frozen checkpoint revision is `signal-checkpoint-v1-4a6b4d974722c9f8729a90d872387bb49e54d01e7bda98ddb69232c28604390e`. Issue #7 may therefore begin from this checkpoint under the freeze and partition rules above once Issue #6's reviewed branch is merged.
