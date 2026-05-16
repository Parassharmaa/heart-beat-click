# H4 Plan — Flow-Matched Cursor > Discrete Click

Date: 2026-05-16. Selected hypothesis from ABSTRACT.md / EXECUTIVE_SUMMARY.md. Cheapest, sharpest, foundational.

## Frozen hypothesis spec

**H4-formal:** On Motor-Bench's trajectory-bound subset, models with a flow-matched continuous cursor head achieve mean task-score ≥ 30 percentage points higher than identical-backbone models with discrete `click(x,y)` / `drag(x1,y1,x2,y2)` action heads, at parity compute, parity data, parity backbone.

- **Primary metric:** task-pass@1, averaged over 5 trajectory-bound families.
- **Secondary:** trajectory IoU, endpoint distance (px), timing error (ms).
- **Falsification:** gap < 10 pp → H4 dead; action head is not the trajectory bottleneck.
- **Ambiguity zone:** 10–30 pp → H4 weakly supported; reframe as H4' "head matters but other factors dominate". Inspect: is the rest of the loop (perception, planner) the bottleneck?

## Fine-tune stack

**Self-managed fine-tune via Hugging Face Transformers + PEFT (LoRA)**, no managed platform. Reasons: custom flow-matching head needs raw PyTorch control over forward, loss, and head registration; LoRA-only managed platforms cannot host an attached non-LoRA head; cheaper at our scale.

- Training framework: `transformers` + `peft` + `accelerate` + `trl.SFTTrainer` (or hand-rolled trainer for the F variant).
- Quantization for memory: `bitsandbytes` 4-bit / 8-bit on the backbone; flow head trained in bf16.
- Data pipeline: `datasets` streaming, on-disk Arrow/Parquet for the cursor-trace JSONL.
- GPUs: rented H100 80GB on **Lambda** (~$2.50/hr), **RunPod** (~$2.00/hr), or **Modal** (~$4/hr but pay-per-second, slick dev loop). Pick by W1.

**Primary backbone:** **Qwen3.5-4B** (Apache-2.0, multimodal — image/video/text, agent-tuned with native tool calling and thinking mode, 262k native context extensible to 1M via YaRN, hybrid Gated DeltaNet + Sparse-MoE architecture). https://huggingface.co/Qwen/Qwen3.5-4B

**Scaling-check backbone:** **Qwen3-VL-30B-A3B-Instruct** (MoE, 3B active, FP8). Tests whether head effect survives at MoE scale. Single seed.

**Out of scope for H4 v0:** Qwen3-VL-235B-A22B (overkill, defer to v1).

**External reference baselines (no fine-tune, eval only):** UI-TARS-7B and UI-TARS-72B (Alibaba, purpose-built GUI VLA, run via HF Transformers outside Tinker since not on Tinker's model list), Microsoft Fara-7B (Qwen2.5-VL-7B-derived, public HF weights), Claude Sonnet 4.6 with computer-use tool, OpenAI Operator. Run as-is on Motor-Bench; report scores for context only — they do not gate any H4 conclusion since they confound architecture and training data.

## Two head variants under test

Both heads consume the same backbone hidden state. Both trained on the same trajectories. Both eval same way.

### Variant D — Discrete (baseline)
- Output vocab adds: `CLICK(x,y)`, `DRAG(x1,y1,x2,y2,duration)`, `SCROLL(dx,dy)`, plus key/event tokens.
- Coordinates discretized to a 1920×1080 grid (or proportional). Standard Fara / Operator / UI-TARS approach.

### Variant F — Flow-matched continuous
- New action head: 4-layer MLP + rectified-flow decoder, predicts the next 1 s of cursor trajectory as a 60-step (dx, dy) sequence at 60 Hz.
- Click / drag implied by trajectory + cursor button-state channel (separate small discrete head).
- Trained with flow-matching loss against ground-truth contractor cursor trace. FAST-DCT compression as an alternative encoding to ablate.
- Inference: emit the 1 s chunk, execute it on the OS while the next forward pass kicks off (RTC-style real-time chunking).

### Streaming input regime (applies to both variants)

Both heads are trained with the model consuming inputs as a **causal stream**, not as single-turn screenshot prompts. Each training trajectory is one long sequence:

```
[task_desc] [tick_0_screen_patches | cursor_state] [tick_0_action_target]
            [tick_1_screen_patches | cursor_state] [tick_1_action_target]
            ...
            [tick_N_screen_patches | cursor_state] [tick_N_action_target]
```

- Screen patches re-encoded incrementally per tick; backbone KV is reused across ticks within a trajectory (streaming session).
- Loss masked to action targets only (action tokens for D, flow-matching loss for F); screen tokens are conditioning context.
- Tick interval at train time matches inference (100 ms). Demos resampled from raw 60Hz cursor + 30FPS screen to 10Hz tick.
- This forbids the half-duplex shortcut of "one screenshot per task" — both variants must integrate evidence across ticks.
- Implementation: pack each trajectory into one HF `datasets` example with per-position attention mask + per-position loss mask; train with `packing=False` (one trajectory per sequence) to keep KV streams clean.

### Held constant
- Backbone weights (init from the same checkpoint).
- Optimizer, LR schedule, total training tokens, total wall-clock.
- Streaming regime (tick-aligned, causal KV reuse).
- Training data (same contractor + synthetic mix).
- Evaluation protocol (same 50 Motor-Bench tasks, same verifiers).

## Motor-Bench (the H4 eval suite)

Subset of HBC-Bench targeting trajectory-bound families only. 50 tasks, 10 per family.

| Family | Verifier | Notes |
|---|---|---|
| Signature drawing (PDF / web) | Trajectory IoU vs target glyph ≥ 0.8 | Free-hand, no snap-to-grid |
| Slider live-preview | Final slider value within ε of target; histogram-cross verifier | Live preview blocked for D variant; F can adapt mid-drag |
| Free-hand region select (Figma / Photoshop) | Selected polygon IoU vs target ≥ 0.7 | Lasso tool |
| Drag-with-autoscroll (Trello / long list) | Target item lands in target list-position | List length 80+, autoscroll engages |
| Map pan-and-zoom (Google Maps) | All N target landmarks visible at zoom ≥ z | Smooth scroll allowed |

Each task: programmatic verifier, time cap 10 s, screencast + cursor trace logged. Stored under `motor-bench/tasks/{family}/{id}.json` with `init_state`, `goal_state`, `verifier.py`.

Tasks generated procedurally where possible (50k random signatures, 50k slider targets), curated where not (the 10 Maps + 10 Trello are hand-picked to vary list length and zoom range).

## Data plan

**Contractor demos**: 2 contractors × 80 hr × ~30 tasks/hr = ~4,800 demos. Each demo is 5–15 s; full session ~100k labelled ticks at 100 ms.

**Synthetic demos**: 10k programmatic expert demos per procedurally generated family. Cheap, used to bias the model toward smooth trajectories.

**Total**: ~50k labelled trajectories, ~20 GB after compression.

**License**: contractor data CC-BY-NC for v0; synthetic released CC-BY. Public release of Motor-Bench separate from training data.

## Compute budget (self-managed HF / rented H100s)

Stage 1 SFT only (no S0 pretrain, no S2 dream-RL — out of scope for H4).

Token volume per SFT run, from contractor data: ~4,800 demos × ~10 s × 10 ticks/s × ~280 tokens/tick ≈ **400 M tokens per epoch**; 3 epochs ≈ 1.2 B training tokens per run.

H100-hour estimates (Qwen3.5-4B LoRA SFT at bf16, batch ~64k tokens, MFU 0.3):
- 1.2 B tok / (40 K tok/sec sustained on 1× H100) ≈ 8.3 H100-hours per run.

- **Qwen3.5-4B SFT** × 2 heads × 3 seeds = 6 runs × ~8 H100-hr × $2.50/hr (Lambda) ≈ **$120**.
- **Qwen3-VL-30B-A3B SFT** × 2 heads × 1 seed = 2 runs × ~40 H100-hr × $2.50/hr ≈ **$200** (scaling check).
- Data pipeline / smoke tests / failed runs buffer: ~$200.
- Eval throughput (50 tasks × 8 models × 3 seeds × real-browser executor): ~$300 (browser VMs > GPU here).
- Contractor demos: 2 contractors × 80 hr × $50/hr ≈ **$8k** (dominant line item).

**Total v0 H4 budget:** ~$8.8k. Contractor labor is the bulk; raw GPU compute is <$1k.

Alt path with more GPU + less data: use VideoAgentTrek's pseudo-labeled corpus + procedural synthetic demos instead of contractors → drop labor cost to near-zero, GPU cost stays ~$1k, total ~$1.5k. Trade-off: noisier action labels. Decide W1 after pilot smoke test.

## Four-week timeline

| Week | Deliverable |
|---|---|
| **W1** | (a) Motor-Bench task spec, 50 verifiers written + tested in sandbox VMs. (b) Recorder stood up: 60 Hz cursor + 30 FPS screen + key events synced, saving to JSONL. (c) Contractor onboarding. |
| **W2** | (a) Collect ~3,000 contractor demos. (b) Implement Variant D and Variant F heads on UI-TARS-7B. (c) Plumb Tinker (or fallback) training loop. (d) First overfit test on 10 demos — both heads should reach 100% on the train set. |
| **W3** | (a) Full SFT runs: 3 seeds × 2 heads × UI-TARS-7B. (b) Eval on Motor-Bench. (c) Run external baselines (Fara-7B, Sonnet 4.6 computer-use, Operator, UI-TARS unmodified). (d) First numbers in. |
| **W4** | (a) Scaling check: same protocol on Qwen3-VL-30B-A3B. (b) Cursor-encoding ablation (FAST-DCT vs rectified-flow). (c) Write 4-page workshop paper. (d) Release Motor-Bench as standalone benchmark. Decision gate: gap ≥ 30 pp → scale to full HBC; gap < 10 pp → reframe; in between → H4'. |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Tinker doesn't support multimodal yet | Fall back to Unsloth / HF Transformers / LLaMA-Factory; budget unchanged. |
| UI-TARS-7B too GUI-specialized → masks head effect | Run Qwen3-VL-30B-A3B in parallel as scaling check. |
| Discrete-head baseline is too weak (apples-to-oranges) | Use Fara-7B / UI-TARS unmodified scores as external reference; if our Variant D underperforms them, training pipeline is broken, not H4. |
| 50 tasks too few for statistical power | Pre-register: paired t-test per family, Bonferroni-corrected; aggregate-mean primary, family-wise secondary. |
| Contractor demos noisy (mouse jitter, hesitations) | Smoothing + dropout on input cursor; flow-matching loss is robust to small label noise. |
| Operator / Sonnet 4.6 numbers unreproducible | Cite once; do not gate any H4 conclusion on closed-source baselines. Open-weight baselines (UI-TARS unmodified, Fara-7B) are the load-bearing reference. |

## Definition of "done" for H4

- Motor-Bench released (public GitHub + leaderboard scaffold).
- Two head variants SFT-trained on UI-TARS-7B (3 seeds each, 95% CI per metric).
- Scaling check on Qwen3-VL-30B-A3B.
- One ablation: rectified-flow vs FAST-DCT cursor encoding.
- 4-page workshop write-up with all five baselines.
- Decision gate result (scale / reframe / kill) committed to repo.
