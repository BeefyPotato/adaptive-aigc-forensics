# Track 5 experiment manifest

Issue #3 implements the reproducible Track 5 source-selection and corruption harness. It includes a content-addressed SID_Set candidate downloader, but downloaded images, inventories, and generated artifacts remain local and are not committed. The tracked `metadata/sid-set-candidate-pool-v1.json` and `metadata/sid-set-candidates-v1.jsonl` files contain only reproducibility metadata, not image bytes.

## Install and controlled fixture

Use Node.js 22 or newer and the checked-in package lock:

```shell
npm ci
npm run track5:fixture
npm test
```

The fixture inspects two checked-in SVG source images and writes deterministic artifacts to `artifacts/track5-fixture/`. Its explicit one-source-per-class split plan is a pipeline check, not an experiment result. Omitting `--split-plan` selects the production allocation and rejects an incomplete inventory.

## Inventory contract

The input is JSON Lines with one object per SID_Set source image:

```json
{"img_id":"stable-SID-id","image_path":"train/real/example.jpg","label":0,"dataset_split":"train","byte_length":12345,"exact_sha256":"64-lowercase-hex-digits","provenance":{"source_dataset":"upstream collection","source_reference":"stable upstream id","license":"declared license"}}
```

- `img_id` must remain stable within the pinned SID_Set revision.
- `image_path` is relative to `--dataset-root`; absolute paths and traversal outside that root are rejected.
- `label` follows SID_Set: `0` authentic, `1` fully synthetic, and `2` tampered. Label `2` is inspected but cannot enter a Track 5 partition.
- `dataset_split` is the upstream `train` or `validation` split.
- Production inventories include the pinned `byte_length` and `exact_sha256` together. The inspector rechecks both immediately before decoding so a file changed after download verification cannot enter selection. Small controlled fixtures may omit both fields.
- provenance requires `source_dataset`, `source_reference`, and `license`.

The inspector decodes every image, records its orientation-corrected dimensions, hashes the original file with SHA-256, and derives a 64-bit difference hash for perceptual leakage checks. These computed fields, the pinned dataset revision, and provenance are embedded in each selected source record.

## Production command and allocation

Download the pinned candidate pool from the repository root:

```shell
node ./scripts/download-sid-set-candidates.mjs
```

The tracked candidate contract freezes the result of scanning SID_Set revision `dc03ead57929879319ce30a82bfcfb8d317b10bd`, letting validation retain any stable identity duplicated in training, and ranking identities with split seed `17`. It records the upstream split, row, stable image ID, label, relative path, byte length, and SHA-256 for 150 reserve candidates per upstream split and class: 14,600 candidates in total.

Hugging Face's rows API does not accept a dataset revision, so the downloader treats its signed image URLs only as an untrusted transport. It accepts a page only when it is complete and every requested row still has the pinned index, image ID, and label; it accepts a downloaded or existing file only when its byte length and SHA-256 match the tracked contract. Changed upstream rows or bytes therefore fail closed. Permanent HTTP errors are not retried, while rate limits, server failures, and network failures use one bounded retry policy.

The command writes the verified local inventory to `datasets/sid-set/inventory.jsonl`. A teammate may populate `datasets/sid-set/images` from a zip or another local mirror first and then run the same command; matching files are verified and reused without network access. The reserve lets collision-aware production selection backfill after exact or perceptual duplicate detection, and the production command fails rather than relaxing leakage rules if the reserve is exhausted.

Then run:

```shell
node ./src/track5-cli.js build-manifest \
  --inventory ./datasets/sid-set/inventory.jsonl \
  --dataset-root ./datasets/sid-set/images \
  --dataset-revision 'saberzl/SID_Set@dc03ead57929879319ce30a82bfcfb8d317b10bd' \
  --output-dir ./artifacts/track5-production
```

Until the organizer archive is available, the audit records its organizer check as `not-available`. Once the official archive has been hashed, add `--organizer-hashes ./datasets/organizer-demonstration-hashes.json`; overlap then becomes a hard failure.

The production defaults use split seed `17`, corruption seed `23`, a perceptual distance threshold of `4`, and exactly this source-level allocation:

| Project partition | Upstream split | Authentic | Fully synthetic | Total |
| --- | ---: | ---: | ---: | ---: |
| Expert training | train | 4,000 | 4,000 | 8,000 |
| Fusion training | train | 1,000 | 1,000 | 2,000 |
| Internal validation | validation | 1,000 | 1,000 | 2,000 |
| Sealed internal test | validation | 1,000 | 1,000 | 2,000 |

Selection hashes the split seed and stable source identity, so inventory input order does not affect the result. The selector protects validation and sealed-test candidates before assigning training candidates. It skips exact duplicates and perceptual matches within distance `4` whenever the match belongs to another project partition, then deterministically backfills from the reserve while preserving every split/class quota. This behavior is recorded as `track5-source-selection-v2` in the manifest. Sources are partitioned before corruption recipes are created, so every clean and corrupted observation from a source inherits one partition. Insufficient collision-free candidates, repeated identities, missing provenance, invalid paths, tampered labels in the selected population, or a failed leakage audit stop the command before artifacts are written.

`track5-manifest.json` contains 14,000 source records and, for a complete production inventory, 280,000 observation records. Each observation links its source and records its binary label, partition, family, severity, deterministic seed, corruption parameters, transform implementation version, and restored dimensions. `track5-leakage-audit.json` is also written separately.

## Corruption conditions

Every source has one clean observation and the following symmetric class-independent variants:

| Family | Conditions | Implementation contract |
| --- | --- | --- |
| JPEG | quality 90, 70, 50, 30 | Actual JPEG encode/decode round trip with 4:2:0 chroma subsampling. |
| Gaussian blur | sigma 0.5, 1, 2 | Explicit deterministic kernel with radius `ceil(3 × sigma)`. |
| Resize | factors 0.5, 0.25 | Lanczos-3 antialiased downscale, then bicubic restoration. |
| RGB Gaussian noise | sigma 0.02, 0.05, 0.10 | Per-channel hash-derived normal noise in RGB `[0,1]`, followed by clamping. |
| Atomic color | brightness, contrast, or saturation × 0.8 or 1.2 | Exactly one named property changes in each variant. |
| Center crop | retain centered 80% | Rounded crop geometry, then bicubic restoration. |

The runtime records exact Sharp and libvips versions. `sharp` is pinned in `package-lock.json`, and the tests inspect encoded JPEG metadata, blur behavior, resize dimensions and kernels, noise repeatability and clamping, atomic color parameters, and crop geometry.

## Leakage and organizer policy

The audit checks exact SHA-256 duplicates and perceptual-hash neighbors across every project partition. If organizer hashes are provided, both checks also run against the organizer demonstration collection. The optional file is a JSON array:

```json
[
  {
    "image_id": "stable-organizer-id",
    "collection": "COCO val2017",
    "exact_sha256": "64-lowercase-hex-digits",
    "perceptual_hash": "16-lowercase-hex-digits"
  }
]
```

When the organizer archive is unavailable, the audit records `not-available` rather than pretending it was checked. Once hashes are supplied, any overlap fails artifact generation. Organizer COCO/DALL-E images are always marked evaluation-only and are prohibited from training, calibration, model selection, and threshold fitting.

## Balanced training sampler

`sampleBalancedTrainingObservations` accepts only the expert-training set or fusion-training set. It allocates each class equally, then sources within each class, then all seven buckets (clean plus six corruption families) within each source, and finally severities within each source/family bucket. Seeded cyclic allocation makes indivisible finite draws differ by at most one while preserving exact class-wide family and severity totals. Internal validation set, sealed internal test set, tampered, and organizer demonstration sources are rejected.
