# FRIDA-Decisions

oMLX serves FRIDA-Decisions through `POST /v1/systemone`. FRIDA evaluates Russian text without generating a reply. It supports `choice`, `score`, `noul` (yes/no), and `ranking` questions.

The adapter uses [FRIDA-Decisions 0.4.0](https://github.com/ai-forever/FRIDA-Decisions/tree/00c8b5f0a88312e969d65b01a076bd9e447de2db), commit `00c8b5f0a88312e969d65b01a076bd9e447de2db`. It loads the original Hugging Face weights with the native MLX backend. It does not need weight conversion, Torch, ONNX Runtime, or vLLM. oMLX retains MLX 0.32.3. Update FRIDA through a deliberate dependency change, then repeat compatibility and parity tests.

## Download and discovery

In the admin Downloader, enter `ai-forever/FRIDA-Decisions`. The Downloader saves the files under `<model-dir>/ai-forever/FRIDA-Decisions` and refreshes the model pool after completion. Its existing filters also download `onnx/model_int8_pertoken.onnx`. The MLX adapter does not use that artifact. The full download needs about 2.91 GB of disk space.

Discovery identifies the files, so renamed local directories and Hugging Face cache snapshots also work. The checkpoint must declare `model_type: "t5"` in `config.json`. It must contain `model.safetensors`, `head.safetensors`, `decisions_config.json`, and `tokenizer.json`. oMLX classifies the checkpoint as a decision model. It excludes decision models from chat selection. Chat endpoints direct callers to `/v1/systemone`.

For repeatable tests, download the validation revision into a local directory:

```sh
hf download ai-forever/FRIDA-Decisions \
  --revision 3a7d751fc7d96c144ff1355d69516b68d751a6f7 \
  --local-dir /path/to/models/ai-forever/FRIDA-Decisions
```

## Ranking request

Use the model ID from the admin model list. Ranking accepts an object with caller IDs, or a list that receives string IDs `"0"`, `"1"`, and so on. It needs at least two non-null candidates. Candidate text can contain JSON objects or arrays. FRIDA renders JSON in a compact form with sorted keys. Candidate IDs do not reach the encoder.

```sh
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "FRIDA-Decisions",
    "state": "Хочу перейти к вам и сохранить свой номер телефона.",
    "questions": {
      "best": {
        "type": "ranking",
        "instructions": "Какие ответы помогут клиенту?",
        "criteria": {
          "port": "Подайте заявку на перенос номера. Номер сохранится.",
          "refund": "Для возврата денег нужен чек.",
          "subscribe": "Подключите платную подписку."
        }
      }
    }
  }'
```

Add your API key header if authentication is enabled. The response keeps the `model`, `answers`, and `usage` fields. Each ranking answer contains `type`, `ranking`, `scores`, `probabilities`, and `confidence`. `ranking` lists the candidate IDs from best to worst. `scores` contains the raw margins, which are the model's candidate scores. Exact ties use string ID order. The adapter preserves upstream numbers without rounding them.

You can mix all four question types in one request. FRIDA uses upstream rules for each question's criteria. Clef and OpenJev reject `ranking` with HTTP 400 before loading weights. FRIDA rejects nonempty `images` with HTTP 400. Invalid criteria return HTTP 400. Invalid shared request types return HTTP 422.

## Precision and memory

FP32 uses 32-bit encoder weights and is the default. BF16 uses 16-bit encoder weights to reduce memory use. Both modes keep the decision head in FP32. Upstream reports 735/735 matching reference decisions in MLX FP32 and 733/735 in BF16. BF16 can change an answer.

For FRIDA, the admin model configuration exposes `frida_precision` with values `fp32` and `bf16`. The value persists for each model. A change triggers the existing safe reload flow. Active requests finish before reload. Pinned FRIDA reloads also pass memory admission checks.

The released encoder contains about 1.65 GB of BF16 weights. FP32 needs about 3.29 GB for those weights. oMLX estimates runtime storage from the safetensors tensor shapes before allocating weights. It includes the FP32 head and a 5% allowance for runtime buffers. Admission also reserves source weights that can coexist with converted weights during loading. These conservative estimates are about 3.22 GiB resident and 4.83 GiB during loading for FP32. BF16 estimates are about 1.61 GiB resident and 3.22 GiB during loading. GiB means 1,073,741,824 bytes. Leave extra memory for encoder attention, request data, other models, and the server.

EnginePool loads FRIDA on the first request and applies normal leases, pinning, and eviction. The lease prevents eviction during execution. All MLX loading, forwards, synchronization, and cleanup use the global MLX executor. The engine yields the GPU after each complete encoder row. It never splits bidirectional T5 attention into causal prefill chunks.

A single-row request uses packed execution, which places its segments in one encoder row. A request with multiple packed rows uses an upstream state cache within that request. The cache stores encoded state tensors and has a 512 MiB limit. It is never shared across requests. Completion, failure, cancellation, and shutdown release it. Shutdown also releases weights and compiled encoder callables.

## Token limits and confidence

With `truncate: true`, FRIDA cuts segments at the upstream limits: 384 state tokens, 96 instruction tokens including the type suffix, and 255 option tokens plus EOS. A token is one unit from the model's tokenizer. EOS is the token that closes an option. With `truncate: false`, any segment that exceeds its limit returns HTTP 413. These limits also apply to rendered JSON text.

`usage.input_tokens` counts encoder tokens actually processed, including repeated state or question segments. It excludes padding. Cached execution counts the state once. `usage.output_tokens` is always zero. FRIDA also returns `usage.state_tokens` and `usage.state_truncated`.

Confidence calculations differ across model families. FRIDA uses one minus normalized entropy, which measures how concentrated the probability distribution is. A uniform distribution gives zero confidence. Clef uses the maximum probability. OpenJev keeps its existing formulas. FRIDA `noul` returns its yes probability without a separate confidence field. Ranking probabilities are relative shares among the supplied candidates. They do not measure each candidate's independent relevance.

BF16 packed and cached execution can produce different margins in upstream 0.4.0. The adapter follows upstream's path selection and preserves its results. Validate each precision separately. Do not require BF16 decisions to equal FP32 decisions.

## Validation

Run the fast adapter tests with `python -m pytest tests/test_frida.py tests/test_systemone.py`. They use tiny original-format checkpoints. The numerical tests share the existing GPU worker group. They cover discovery, precision, memory admission, row scheduling, leases, eviction, reload, errors, and cleanup.

To run real-weight tests on Apple Silicon, set an explicit local model path:

```sh
OMLX_FRIDA_MODEL_PATH=/path/to/models/ai-forever/FRIDA-Decisions \
  python -m pytest tests/test_frida_integration.py \
  -m 'slow and integration' -s
```

These tests skip when the local weights are absent. They compare the adapter against the pinned upstream release separately for FP32 and BF16. Adapter-to-upstream margins use `atol=1e-3, rtol=1e-3`. Tests require matching decisions and full ranking order. Packed/cache comparisons require matching decisions and ranking order at both precisions. FP32 margins use the same tolerance. Tests report BF16 packed/cache margin drift because upstream itself does not meet that numerical tolerance. They cover Russian text, all four question types, multiple rows, JSON states, and truncation.

See [the implementation validation report](FRIDA_VALIDATION.md) for measured memory and test results. Fast CI does not replace real-weight tests and a Downloader smoke test on Apple Silicon.
