# Adaptive AIGC Forensics

This context covers image-level detection of AI-generated content when image evidence may be weakened by real-world transformations.

## Language

**RGB expert**:
An authenticity detector that reasons from the normally decoded RGB image.
_Avoid_: Semantic expert, visual model

**Signal expert**:
An authenticity detector that reasons from explicit low-level image statistics rather than another learned RGB representation.
_Avoid_: Frequency model, second vision backbone

**Degradation gate**:
An estimator that assigns per-image trust between experts from observed image-quality evidence; it does not independently decide whether an image is authentic or AI-generated.
_Avoid_: Degradation classifier, authenticity gate

**Complementary value**:
Evidence that one expert corrects errors made by another on held-out conditions, rather than merely duplicating its predictions.
_Avoid_: Feature importance, ensemble gain

**Source image**:
The original dataset image that owns every clean or transformed observation derived from it and therefore determines their shared data split.
_Avoid_: Original sample, parent image

**Corruption variant**:
A deterministic observation derived from a source image using a declared real-world transformation, severity, and seed.
_Avoid_: Augmented image, transformed copy

**Materialized observation**:
The lossless RGB pixel artifact produced by resolving one corruption variant, shared by expert branches before their branch-specific preprocessing.
_Avoid_: Corrupted source, augmented file

**Corruption harness**:
The reproducible system that creates or describes clean and corrupted observations symmetrically across authenticity classes for training and evaluation.
_Avoid_: Augmentation pipeline, preprocessing

**Signal representation**:
The fixed-length deterministic description of low-level evidence supplied to the signal expert.
_Avoid_: Signal image, forensic embedding

**Experiment profile**:
A named, provenance-bound selection of observations and resource limits for the same scientific pipeline. The `hackathon-v1` profile changes sample counts only; it does not change source partitions, corruptions, features, model architecture, leakage rules, or evaluation semantics.
_Avoid_: Different model, relaxed pipeline, pilot system

**Residual kernel**:
The fixed internal smoothing operation used to isolate fine-detail evidence from luminance; it is not a real-world corruption.
_Avoid_: Blur augmentation, Gaussian corruption

**Internal validation set**:
The source-disjoint development partition used for model selection, calibration, and threshold selection.
_Avoid_: Dev test, organizer validation

**Fusion-training set**:
The source-disjoint training partition used to calibrate frozen expert outputs and train static or adaptive fusion without reusing signal-expert training sources.
_Avoid_: Internal validation set, gate validation

**Sealed internal test set**:
The source-disjoint partition withheld from all model and threshold decisions and used once for final internal reporting.
_Avoid_: Final validation, holdout validation

**Organizer demonstration set**:
The organizer-provided COCO val2017 and DALL-E Advanced reference benchmark that is evaluation-only and does not influence training or model selection.
_Avoid_: Validation set, training benchmark

**Mean corrupted AUROC**:
The mean of the six corruption-family AUROCs after averaging severities within each family.
_Avoid_: Overall AUROC, augmented AUROC

**All-condition macro AUROC**:
The primary summary metric that gives equal weight to clean performance and each of the six corruption-family results.
_Avoid_: Mean corrupted AUROC, pooled AUROC

**Static fusion**:
An expert combination that uses the same learned trust allocation for every image condition.
_Avoid_: Degradation gate, fixed 50/50 fusion

**Quality evidence**:
Measurements describing how image evidence may have been degraded without asserting whether the image is authentic or AI-generated.
_Avoid_: Authenticity evidence, corruption label
