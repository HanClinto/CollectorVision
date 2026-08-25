---
library_name: onnxruntime
tags:
  - collectorvision
  - model-registry
---

# CollectorVision Artifact Registry

This repository is the canonical registry for CollectorVision model artifacts.
It contains metadata only; model binaries remain in their architecture-family
repositories, such as `HanClinto/cornelius` and `HanClinto/milo`.

`registry.json` contains exact model records with immutable Hugging Face commit
revisions, file names, byte sizes, and SHA-256 digests. Its `stable` and
`testing` channels map a model family to a supported exact model ID.

Clients may cache the registry and refresh it periodically. They must download
model artifacts from the recorded immutable revision and verify the SHA-256
digest before using the file.