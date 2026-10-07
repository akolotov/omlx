# FRIDA implementation validation

The implementation was tested on 2026-10-07 with an Apple M4 and 32 GiB of unified memory. The tests used Python 3.12.12 and MLX 0.32.3. FRIDA used release 0.4.0, commit `00c8b5f0a88312e969d65b01a076bd9e447de2db`. Original validation weights used Hugging Face revision `3a7d751fc7d96c144ff1355d69516b68d751a6f7`.

## Tests and parity

The full CI command passed with 16,585 tests passed and 1,261 skipped. The command was `python -m pytest tests/ -m "not slow and not integration" -n 3 --dist loadgroup`. The targeted decision, API, discovery, configuration, profile, and pool run passed all 1,138 tests. Ruff and `git diff --check` passed for the new adapter and tests.

The localization follow-up passed 443 targeted tests and three JavaScript dashboard tests. FRIDA precision labels render from the translation catalogs in all ten locales. `scripts/normalize_i18n.py` and the CSS rebuild completed. Black formatting passes for the added Python code relative to upstream `main`.

Both real-weight test cases passed, one for FP32 and one for BF16. Each precision covered four requests and ten decisions. The requests included Russian text, all four question types, JSON states and criteria, multiple rows, state caching, and truncation. Adapter margins matched the corresponding upstream execution path with `atol=1e-3, rtol=1e-3`. The largest adapter/upstream margin difference was `4.76837158203125e-7` in FP32 and `0.0` in BF16. Decisions and complete ranking orders matched upstream.

Packed and cached execution produced the same decisions and complete ranking orders in these cases. The largest margin difference was `1.0967254638671875e-5` in FP32. The BF16 difference was `0.17119884490966797`. The pinned upstream itself produces this BF16 difference. BF16 packed/cache margins therefore do not satisfy `atol=1e-3, rtol=1e-3`. The tests retain that tolerance for adapter/upstream comparisons. They report the BF16 packed/cache margin difference and require matching decisions and ranking order. This distinction is a numerical limitation of the selected upstream release.

FP32 and BF16 produced the same ten decisions in this test set. Their numerical scores differed. This small test set does not establish equality across precisions. Upstream reports 733/735 matching reference decisions for BF16, so BF16 can change an answer.

## Memory and Downloader smoke test

The measured active weight storage was 3,293,625,348 bytes (3.07 GiB) in FP32. The FP32 loading peak was 4,940,437,912 bytes (4.60 GiB). BF16 active weight storage was 1,646,815,748 bytes (1.53 GiB). Its loading peak was 1,646,818,712 bytes (1.53 GiB). These are MLX active allocations, not the total process footprint. The BF16 estimate conservatively charges source and runtime weights separately even when MLX shares their buffers.

EnginePool committed 3,458,291,563 bytes for FP32 and 1,729,149,009 bytes for BF16. These estimates include the FP32 head and runtime allowance. Admission reserved the larger temporary loading estimate before weight allocation. Fast pool tests also covered insufficient memory, pinned protection, active leases, deferred precision reload, idle eviction, and committed storage after unload.

The unchanged HFDownloader downloaded `ai-forever/FRIDA-Decisions` in 40.5 seconds. It included `onnx/model_int8_pertoken.onnx`. Its completion callback discovered the checkpoint as `decision`, with engine type `decision`. The download metadata recorded the pinned validation revision.

The Apple Silicon smoke test used the downloaded directory through the real EnginePool. It answered ranking and mixed questions in default FP32, unloaded, and loaded again on another request. It repeated that sequence with explicit BF16. Both precisions ranked the example IDs as `port`, `subscribe`, `refund`. Each request processed 291 encoder tokens. After each unload, committed pool memory was zero and active MLX storage was 1,040 bytes. Reload preserved the decisions and ranking order. The example showed no decision difference between precisions.

## Dependency and bundle checks

A clean oMLX environment resolved the new core dependency with the existing MLX pin. `pip check` reported no broken requirements. The installed FRIDA metadata recorded the full requested Git commit. FRIDA core requirements contain NumPy, tokenizers, safetensors, and Hugging Face Hub. FRIDA did not add Torch, ONNX Runtime, or vLLM. Existing oMLX dependencies independently install ONNX Runtime.

The generated bundle requirements included the full FRIDA pin. The packaging pipeline built `frida_decisions-0.4.0-py3-none-any.whl` and mapped the generated requirement to that wheel. The wheel contains `frida_decisions/mlx_backend.py` and `frida_decisions/mlx_modeling.py`.

The full venvstacks build, lock, and export completed. The generated framework layer imports FRIDA 0.4.0 under Python 3.11.10 and retains MLX 0.32.3. Both MLX backend modules are present in `packaging/_export/framework-mlx-base/lib/python3.11/site-packages/frida_decisions`.

The final Swift `.app` build could not run because Xcode is absent. The host has Command Line Tools only. The runtime export is complete, but inclusion in a complete generated `.app` remains unverified.
