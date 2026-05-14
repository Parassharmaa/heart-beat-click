# Heart Beat Click — Compute & Latency Budget

Date: 2026-05-14. Goal: sanity-check whether 100 ms tick on a 7–14B foreground is plausible, grounded in Moshi / TML / OpenVLA-OFT / PD-VLA measurements.

## 1. Per-tick token budget

| Stream | Rate | Tokens / tick (100 ms) | Notes |
|---|---|---|---|
| Screen patches in | 1 frame / 100 ms (10 FPS) | ~256 patches | 1080p frame → 1920/40 × 1080/40 ≈ 48×27 = 1296 patches; spatial down-pool to 256 |
| Cursor proprio in | 60 Hz raw → 6 / tick | ~6 | (x,y,btn) bucketized |
| DOM delta in | per change | 0–32 | optional aux; usually 0 |
| User text/voice in | sparse | ~0–20 | only when user speaks |
| Cursor head out | continuous chunk | 1 (a 100ms flow-matched trajectory chunk) | π0/RTC style |
| Event head out | discrete | 0–4 | sparse |
| Inner monologue out | text | 0–10 | mostly silent; rare reasoning bursts |
| Dispatch out | discrete | 0–2 | rare |
| Screen patch *prediction* (world-model head, train-only or aux at infer) | optional | ~256 if used | drop at infer for budget |

Steady-state input tokens per tick ≈ **260–280**. Output tokens per tick ≈ **5–15**. Sequence growth ≈ 270/tick → 162k tokens / minute / session. KV bloat is the real cost; sliding window + episodic compression (StreamAgent's hierarchical KV-cache) is mandatory.

## 2. Reality check against Moshi

Moshi 7B sustains TTFA <200 ms on a single H100 at full-duplex audio (12.5 audio tokens/sec across two streams + inner monologue ≈ 25 tok/sec output, with 12.5 tok/sec input chunks). KV grows at ~25 tok/sec.

HBC has ~10× more input tokens/sec (260 vs ~25), comparable output tok/sec, and one extra (small) depth transformer pass per tick. Naive scaling: 10× input → ~2× more KV-bandwidth (input doesn't generate, but does write KV). End-to-end tick budget at 7B ≈ 150–250 ms on a single H100 without further optimization. **Tight at 100 ms; comfortable at 200 ms.**

## 3. TML's optimization stack (which we inherit)

- **Streaming session KV** persistent on GPU, no per-tick realloc. (TML blog.)
- **Gather+gemv MoE kernels** for bidirectional serving. Buys ~2× on MoE foregrounds.
- **Batch-invariant kernels** for trainer-sampler bit match. Cost <5% throughput, big win for RL stability.
- **PagedAttention** + quantized KV (int8 / fp8). Buys ~2× memory and ~1.5× bandwidth.
- **PD-VLA parallel decoding** for action chunks. Up to ~25× on action-token throughput (relevant to dispatch + event + monologue heads).
- **RTC (Real-Time Chunking)** for the cursor head: generate next chunk while executing current → hides one tick of cursor-decode latency.
- **OpenVLA-OFT continuous-action L1**: 25× faster generation than discrete AR action tokens.

With this stack, 100 ms tick at 7B foreground is plausible on 1× H100, 100 ms at 14B requires either 2× H100 or fp8-quantized weights or a sparser MoE.

## 4. Memory

Llama-7B fp16 weights ≈ 14 GB. KV cache fp16 ≈ 2 GB for 4k ctx batch 1. Streaming sessions need 64k+ ctx → ~32 GB KV unless we slide. With sliding-window of last 30 s (300 ticks × ~270 tok ≈ 81k tok) and episodic compression for older context, 80 GB H100 fits batch 1 comfortably, batch 4 with int8 KV.

## 5. Training compute

Stage 0 (screencast world-model pretrain): VideoAgentTrek used **26B tokens** over **39k videos / 1.52M ReAct steps** to get 15.8% OSWorld-Verified — but only as continued pretrain on text-based ReAct steps. HBC's world-model pretrain target is **larger** because we additionally predict frame chunks: budget 100–300B tokens on a 7B base. At MFU 0.4 on H100 80GB ≈ 350 TFLOPS sustained, that's 100B × 6 × 7e9 / 350e12 = 1.2e10 s = **3,300 H100-hours**. Real number ~5,000 H100-hours with overheads. Comparable to a small fine-tune run, not a from-scratch frontier pretrain.

Stage 1 (duplex SFT): smaller, 1–5B tokens of labeled duplex trajectories. ~500 H100-hours.

Stage 2 (dream RL): bounded by browser executor throughput, not GPU. ~1k H100-hours, ~10k executor-hours.

Stage 3 (RLHF / barge-in): small. ~200 H100-hours.

**Total v0 budget**: ~7k H100-hours ≈ $20k–30k at on-demand cloud rates, or ~one month on a 16-GPU H100 cluster. Reproducible outside frontier labs.

## 6. Where the budget breaks

- 14B foreground at 100ms = needs 2-GPU pipeline or 4-bit weight quant. Foreground stays 7B v0.
- 60 FPS cursor head requires the flow-matching decoder to fire 6× per tick. Cheap (small head), but adds ~5–10ms.
- World-model head at inference is the killer (256 patch tokens × autoregressive = ~10× current decode load). Use it at train time + dream rollouts only; drop at live inference (still get the latent forward model via the trunk).
- Dispatch round-trip to background reasoner = 200–1000 ms one-way. Foreground must never block — pure async.

## 7. Latency comparison table

| System | Loop type | Tick / cycle | Source |
|---|---|---|---|
| OpenAI CUA | Half-duplex screenshot | 1–5 s | os-world benchmarks |
| Fara-7B | Half-duplex screenshot | ~16.5 actions / task, seconds each | Fara paper |
| Claude Sonnet 4.6 + computer-use | Half-duplex | ~1–3 s | anecdotal |
| Moshi (speech, 7B) | Full-duplex audio | <200 ms TTFA | Spheron benchmarks |
| TML-Interaction-Small (276B MoE, 12B active) | Full-duplex multimodal | 200 ms micro-turn, 400 ms turn-taking | TML blog |
| **HBC target** | Full-duplex screencast | **100 ms tick, <50 ms cursor reaction** | this doc |

HBC's tick is 2× faster than TML's (because we need motor control granularity, not turn-taking granularity); compensated by being 20× smaller (7B vs 276B MoE).
