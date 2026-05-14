# Heart Beat Click — Paper Outline

Date: 2026-05-14. Target venue: ICLR / NeurIPS systems-and-foundation-models track, or arxiv preprint with workshop submission.

## Length budget

Main paper: 9 pages (NeurIPS / ICLR limit). Appendix unlimited (move ablation details, data pipeline diagrams, benchmark task catalog there).

## Sections

### 1. Introduction (1.25 pp)
- Hook: speech crossed duplex chasm in 2024–2026 (Moshi → PersonaPlex → TML-Interaction); GUI did not.
- OSWorld near saturation (Sonnet 4.6 = 72.5% vs human 70–75%); remaining errors concentrated where observation is the bottleneck.
- Stating problem: half-duplex screenshot loop is structurally limited on time-sensitive UI dynamics.
- Stating contributions:
  1. HBC architecture (one-paragraph summary).
  2. HBC-Bench: 200 task benchmark of duplex-necessary GUI tasks.
  3. Five falsifiable hypotheses; ablations show which earn their compute.
- Figure 1: HBC system diagram, single page.

### 2. Background and related work (1 pp)
- Three columns:
  - Duplex multimodal models: Moshi (RQ-Transformer), PersonaPlex, TML-Interaction-Small (encoder-free + dispatch).
  - Computer-use agents: CUA, Fara-7B, ShowUI, Claude computer-use, ScaleCUA.
  - World models for GUI / control: WebWorld, WebSynthesis, Dreamer 4, WorldVLA, DreamVLA, π0/OpenVLA-OFT, FAST.
- One concise table: what each precedent has and lacks.
- Explicit non-claim of novelty for individual ingredients (sets reviewer expectations).

### 3. Method: HBC architecture (2.25 pp)
- 3.1 Input/output streams (per-tick token budget table).
- 3.2 RQ-Transformer backbone (large temporal + small depth).
- 3.3 Encoder-free early fusion (hMLP screen patches + dMel audio + learned event embeddings).
- 3.4 Heads: (a) flow-matched cursor head, (b) discrete event AR head, (c) inner-monologue text AR head, (d) dispatch AR head, (e) optional screen-prediction head for joint W+P loss.
- 3.5 Joint world-model + policy loss; shortcut-forcing trick from Dreamer 4 for the screen head; FAST-DCT for cursor.
- 3.6 Dual-model dispatch protocol (foreground/background async, non-blocking).
- 3.7 Inference: streaming KV session, sliding window, hierarchical episodic compression.
- Figure 2: per-tick token timeline (left: speech-side Moshi/TML; right: HBC screencast/cursor).
- Figure 3: dispatch protocol sequence diagram.

### 4. Training recipe (1 pp)
- Four stages: S0 screencast world-model pretrain → S1 duplex SFT → S2 dream RL with executor verification → S3 duplex RLHF/barge-in.
- Data tables: VideoAgentTrek (10k hr), MONDAY, ScaleCUA, contractor 60-Hz cursor logs.
- IDM details: extend Video2Action with continuous cursor trace recovery.
- Compute table: ~7k H100-hours total for v0.
- Figure 4: data flow + loss diagram across stages.

### 5. HBC-Bench (0.75 pp)
- Inclusion test (5 criteria: trajectory-bound, mid-action observation, hard timing, continuous feedback, multi-stream).
- Nine families, 200 tasks total, per-family verifier description.
- Metrics: Pass@k, tick latency, score quality (IoU / timing error), mid-action correction rate.
- Figure 5: representative tasks (5–8 panels: signature, drag-autoscroll, slider, video scrub, map pan, reaction toast).

### 6. Experiments and ablations (2 pp)
- 6.1 Baselines: half-duplex SOTA (Claude Sonnet 4.6, Operator, Fara-7B, ShowUI) on HBC-Bench + OSWorld-Verified + WebArena + AndroidWorld.
- 6.2 Core results table: HBC vs baselines across all four eval suites.
- 6.3 Ablation table (A0–A8): every component isolated.
- 6.4 Tick-rate sweep figure: HBC-Bench score vs tick rate.
- 6.5 Cursor-encoding sweep.
- 6.6 Background-reasoner sweep (size, type, dispatch bandwidth vs success).
- 6.7 Wall-clock-per-task plot.
- Figure 6: ablation grid heatmap.
- Figure 7: trajectory comparison (HBC continuous cursor vs Operator discrete drag on a signature task).

### 7. Discussion (0.5 pp)
- Where duplex helps least: tasks fully solvable by discrete planning (form-filling). HBC is not the right tool everywhere.
- Where it helps most: trajectory, timing, mid-action correction.
- World-model-as-policy: gradient sharing implies the latent forward model is policy-aligned in a way an external world model never is.

### 8. Limitations and broader impact (0.25 pp)
- Compute / data assumptions for v0.
- Foreground / background trust boundary; prompt-injection risk surface widens with faster action (see SAFETY.md).
- Surveillance / privacy implications of always-on screencast capture.

### 9. Conclusion (0.5 pp)

### Appendix
- A. Full HBC-Bench task catalog (1 row per task, 200 rows).
- B. Per-stage data pipeline diagrams.
- C. Full ablation results with confidence intervals.
- D. Inference-stack pseudocode (streaming KV session, dispatch protocol).
- E. Failure-case gallery.
- F. Compute / cost breakdown.
- G. Safety / red-team protocol used pre-release.

## Figure list (final 7)

1. HBC system diagram (intro).
2. Per-tick token timeline, speech vs GUI.
3. Dispatch protocol sequence.
4. Training-stage data flow.
5. HBC-Bench representative tasks.
6. Ablation heatmap.
7. Continuous vs discrete cursor trajectory comparison.

## Anonymous URLs

- `hbc-anon.github.io`: paper site, demo videos, code (release on accept).
- `hbc-bench.github.io`: standalone benchmark site, leaderboard, eval harness.

## Open-source plan

- Stage 0 pretraining recipe + IDM code (cursor-trace recovery extension to Video2Action).
- HBC reference implementation in PyTorch, 7B foreground.
- HBC-Bench eval harness + task catalog.
- Withhold Stage 2 dream-RL pipeline initially (high red-team surface), open-source after audit.

## Reviewer-targeting

Anticipated objections + where they're answered:
- "Joint world+policy isn't new" → NOVELTY.md / §2 explicit non-claim + intersection argument.
- "Why not just sample screenshots faster?" → A0-Plus negative control, infinite-screenshot half-duplex baseline.
- "100 ms is unrealistic on 7B" → COMPUTE.md + measured latency table from inheriting TML stack.
- "OSWorld isn't really saturated" → not the wedge; HBC-Bench is, and we report OSWorld too.
- "Safety with faster actions" → §8 + appendix G + revocable-action gating + confirmation-on-irreversible.
