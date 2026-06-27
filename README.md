# dspark-vllm

*A vLLM overlay that adds DeepSeek's DSpark speculative decoding to DeepSeek-V4-Flash on RTX Pro 6000 (Blackwell, sm_120). By [Fraser Price](https://x.com/fraserpricee).*

DSpark (from DeepSeek's [DeepSpec](https://github.com/deepseek-ai/DeepSpec)) is block-wise speculative decoding: each draft step proposes a block of tokens through a small Markov head instead of one token at a time, so it is faster than the model's stock single-layer MTP. No released inference engine ships it yet; this repo adds it to a Blackwell vLLM build as a thin, swappable overlay.

The resulting image is published at [`fraserpricee/vllm:dspark-cu132-20260627`](https://hub.docker.com/r/fraserpricee/vllm) and consumed out of the box by:

- [fraserprice/DeepSeek-V4-Flash-DSpark](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark): stock DeepSeek-V4-Flash + DSpark
- [fraserprice/DeepSeek-V4-Flash-Abliterated-DSpark](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-Abliterated-DSpark): abliterated DeepSeek-V4-Flash + DSpark

Each ships a one-command `run.sh`. You only need this repo to rebuild the image, not to run the models.

## ⚠️ Scope: local 2-4x RTX Pro inference, not high-concurrency serving

This is a local-inference implementation. It deliberately omits one part of full DSpark:

- **No confidence-scheduled verification.** Full DSpark uses the draft's per-token confidence (prefix survival probabilities) to dynamically schedule how many drafted tokens to verify per step, which is a throughput win under high batch concurrency. This overlay loads the confidence head but does not use it for scheduling: it verifies a fixed `num_speculative_tokens` block with standard probabilistic rejection.

For single-stream and low-concurrency local workloads on 2-4 GPUs (the target here) that costs little, keeps the verification path simple, and still beats stock MTP. For high-volume production serving, the upstream DeepSpec scheduler is the better fit.

Speculative decoding is lossless with respect to the target distribution: accepted tokens are exactly the target model's, so quality is unchanged regardless of the draft. The flags and defaults throughout assume RTX Pro 6000 Blackwell (96 GB, sm_120) at `TP=2` or `TP=4`.

## What's in the overlay

`overlay/vllm/` is copied over the base image's `vllm` package (see `Dockerfile`). It adds:

| Area | File | Purpose |
|---|---|---|
| Speculative config | `config/speculative.py` | registers the `dspark` method |
| Draft model | `models/deepseek_v4/nvidia/dspark.py` | `DSparkDraftModel`: block draft (attention + MoE) and Markov / confidence heads |
| Registry | `model_executor/models/registry.py` | maps `DSparkDraftModel` |
| Proposer | `v1/spec_decode/dspark.py`, `dspark_markov.py` | block proposal via `sequential_markov_sample`, fixed-length verify |
| Base proposer | `v1/spec_decode/llm_base_proposer.py`, `dflash.py` | shared draft-runner plumbing |
| Attention | `v1/attention/backends/mla/sparse_swa.py` | sparse MLA / sliding-window path for the draft |

## Build

```bash
./build.sh                              # -> fraserpricee/vllm:dspark-cu132-20260627
./build.sh myname/vllm:dspark-custom    # custom tag
```

The build is `FROM voipmonitor/vllm:eldritch-final-vbfaa36b-b12x284a2ea-kimi-specdcp-cu132-20260627` (a Blackwell / CUDA-13.2 vLLM build that runs the DeepSeek-V4 architecture: sparse attention / lightning indexer, MLA, FP8 experts) and overlays the files above. The `Dockerfile` import-checks every changed module, so a broken overlay fails the build.

## Test

```bash
python -m pytest tests/
```

## Credits

- [DeepSeek-AI / DeepSpec](https://github.com/deepseek-ai/DeepSpec): the DSpark technique and reference implementation.
- [`voipmonitor/vllm`](https://hub.docker.com/r/voipmonitor/vllm): the Blackwell / CUDA-13.2 vLLM base image.
- [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash): the target model.

## License

Apache-2.0 for the overlay code (matching vLLM). The base image and model weights carry their own licenses.
