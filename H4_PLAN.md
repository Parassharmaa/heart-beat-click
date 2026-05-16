# H4 Plan — Flow-Matched Cursor > Discrete Click

Date: 2026-05-16. Selected hypothesis from ABSTRACT.md / EXECUTIVE_SUMMARY.md. Cheapest, sharpest, foundational.

## Frozen hypothesis spec

**H4-formal:** On Motor-Bench's trajectory-bound subset, models with a flow-matched continuous cursor head achieve mean task-score ≥ 30 percentage points higher than identical-backbone models with discrete `click(x,y)` / `drag(x1,y1,x2,y2)` action heads, at parity compute, parity data, parity backbone.

- **Primary metric:** task-pass@1, averaged over 5 trajectory-bound families.
- **Secondary:** trajectory IoU, endpoint distance (px), timing error (ms).
- **Falsification:** gap < 10 pp → H4 dead; action head is not the trajectory bottleneck.
- **Ambiguity zone:** 10–30 pp → H4 weakly supported; reframe as H4' "head matters but other factors dominate". Inspect: is the rest of the loop (perception, planner) the bottleneck?

## Backbone choices

Use Thinking Machines' **Tinker API** as the primary fine-tune platform (GA since 2026; managed distributed training, LoRA, token-based pricing). Tinker's supported model list includes the latest Qwen-VL multimodal MoEs (verified at https://thinkingmachines.ai/tinker/). Tinker constraints to respect: **LoRA-only**, custom loss is plausible (via raw `forward_backward` / `optim_step` primitives in the Cookbook) but unconfirmed in docs — verify in W1.

**Primary backbone:** **Qwen3.5-4B** (Apache-2.0, multimodal — image/video/text, agent-tuned with native tool calling and thinking mode, 262k native context extensible to 1M via YaRN, hybrid Gated DeltaNet + Sparse-MoE architecture). Confirmed on Tinker's price sheet at $0.22/M prefill, $0.67/M train. Smallest viable size; clean isolation of the cursor head; long context handles entire screencast sessions in one window without sliding-window complexity. https://huggingface.co/Qwen/Qwen3.5-4B

**Scaling-check backbone:** **Qwen3-VL-30B-A3B-Instruct** (MoE, 3B active, FP8). Tests whether head effect survives at MoE scale and at a stronger VL-specialized backbone. Single seed only.

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

### Held constant
- Backbone weights (init from the same checkpoint).
- Optimizer, LR schedule, total training tokens, total wall-clock.
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

## Compute budget (Tinker token-priced)

Stage 1 SFT only (no S0 pretrain, no S2 dream-RL — out of scope for H4).

Token volume per SFT run, from contractor data: ~4,800 demos × ~10 s × 10 ticks/s × ~280 tokens/tick ≈ **400 M tokens per epoch**; 3 epochs ≈ 1.2 B training tokens per run.

- **Qwen3.5-4B SFT** × 2 heads × 3 seeds = 6 runs × ~1.2 B tok × $0.67/M ≈ **$4.8k** (primary).
- **Qwen3-VL-30B-A3B SFT** × 2 heads × 1 seed = 2 runs × ~1.2 B tok × $0.40/M ≈ **$1k** (scaling check).
- Eval (Tinker sample API): 50 tasks × 6 models × 3 seeds ≈ negligible (a few $).
- HF fallback (UI-TARS reference + custom-head smoke tests + Tinker compatibility): ~$1k on Lambda / RunPod H100 nodes.

**Total v0 H4 budget:** ~$7k all-in. Single-developer four-week sprint.

If Tinker does not support custom heads (verified W1 via cookbook + `forward_backward` primitive probe), path moves entirely to HF Transformers + Unsloth on rented H100s; budget shifts to GPU-hours: ~400 H100-hours ≈ $1.2k, but loses Tinker's managed-cluster speedup and LoRA infra.

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
