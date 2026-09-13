# Bundled fusion model

`logistic.json` is restored byte-for-byte from commit `2a498a4`. SHA-256: `e9f864956aa65c3f05c038b13b0c8ca9e7289e4a44f6f0eb7837ab85814c730e`.

This is the synthetic-trained 13-feature logistic model selected in the earlier development audit. Its saved threshold is 0.08833933; production explicitly uses 0.15. No weights were refitted during restoration. The code checks the bundled hash and resolves its path relative to `pipeline.py`.

See [current workflow](../../PRODUCTION_WORKFLOW.md) and [restoration receipt](../../docs/fusion-restored-validation.json). Copy this artifact along with the inference source when preparing an offline bundle.
