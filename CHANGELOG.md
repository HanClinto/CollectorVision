# Changelog

## Unreleased

- Default corner detection now uses Cornelius 1.221 from Hugging Face. This fixes the rotated-card quality regression tracked in [issue #24](https://github.com/HanClinto/CollectorVision/issues/24), where soft-argmax averaging could place corners between competing peaks.
- The Python library and generated web scanner assets now share the same bundled `collector_vision/weights/cornelius.onnx` default, so web scanner refreshes will publish Cornelius 1.221 automatically.
