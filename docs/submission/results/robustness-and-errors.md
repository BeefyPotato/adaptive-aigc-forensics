# Robustness and errors

## Clean versus transformed

System: learned-static-fusion. This internal validation report is not an official organizer score.

| Condition | AUROC |
| --- | ---: |
| Clean | 0.981975 |
| JPEG | 0.965512 |
| Blur | 0.975375 |
| Resize | 0.962363 |
| Noise | 0.883883 |
| Color | 0.975696 |
| Crop | 0.977675 |
| Mean transformed | 0.956751 |

## Persisted summary

| Metric | Value |
| --- | ---: |
| All-condition macro AUROC | 0.960354 |
| Brier score | 0.099829 |
| Balanced accuracy | 0.876250 |
| FPR | 0.084000 |
| FNR | 0.163500 |

## Threshold trade-offs

The weakest persisted family/severity is noise / sigma-0.1 (AUROC 0.810425). The persisted threshold trade-off has balanced accuracy 0.876250, FPR 0.084000, and FNR 0.163500. This discussion is descriptive, not causal.

## Error strata

**False positives** and **False negatives** below are deterministic sanitized representatives from the frozen evaluation; they do not establish visual causes.

### Clean False Positive

| Source | Variant | Family | Severity | Error | Expert agreement | Correction status | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sid-set:0bffbd3a7d89efff | variant-v1-cdab6df3ac140c695cb2c316eeac52bafe7438eab0684ac354756eefd8bbf271 | clean | clean | false-positive | disagree | rgb-corrects-signal-error | 0468fa8d8489a1071a677d55fd5c05206b04c6c9a3586d373ac9c70ddc694df3 |

### Clean False Negative

| Source | Variant | Family | Severity | Error | Expert agreement | Correction status | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sid-set:full\_synthetic\_005849 | variant-v1-0982257336b9674738600a3757d0c1e1fe0922ade2db1683403337120e08893d | clean | clean | false-negative | disagree | signal-corrects-rgb-error | 0d19c682e5aef4deca8c9df50864e9583a72aa9b4e4fa93adc56905cd94bd32c |

### Transformed False Positive

| Source | Variant | Family | Severity | Error | Expert agreement | Correction status | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sid-set:01741ff5e11d9c31 | variant-v1-896cd5429fc0b53e05bee9e60a941a752c0027aec8ced72d54ab261a966c42af | blur | sigma-2 | false-positive | disagree | signal-corrects-rgb-error | 000393fb66ad6aa5559269692440bf95c51dd58490b49ee49190de38ca6839c3 |

### Transformed False Negative

| Source | Variant | Family | Severity | Error | Expert agreement | Correction status | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sid-set:full\_synthetic\_005762 | variant-v1-cb2e06d89e9e68053d2d970635056bcf2093b97bd59e9a2f1adc911d491597fc | noise | sigma-0.05 | false-negative | agree | both-experts-wrong | 01333945f0053874e455c4a2f3450153fcc6fe9b32fa301f828a8512789cfcf9 |

## Limitations

- Internal validation influenced candidate and threshold selection.
- These results are not organizer, sealed, official, independent-test, or unbiased estimates.
- Upstream checkpoint overlap cannot be disproven.
- The weakest-condition/FPR/FNR discussion is descriptive, not causal.
