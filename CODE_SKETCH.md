# Heart Beat Click — Backbone Pseudocode

Date: 2026-05-14. PyTorch-style pseudocode for the core `tick()` and training loss. Not runnable; meant to make the architecture concrete enough to spot gaps.

## 1. Stream / token layout

```python
# Per-tick tensors (B = batch, T_in = input tokens this tick, K = codebook channels)
# All time-aligned: same tick index for all streams.

ScreenPatches : Tensor[B, N_patch, D_patch_in]   # ~256 patches per tick (hMLP input)
CursorProprio : Tensor[B, 6, 3]                  # 6 raw mouse samples / 100 ms; (x_norm, y_norm, btn_state)
DomDelta      : Optional[Tensor[B, N_dom, D_dom]]  # variable, often empty
UserText      : Optional[Tensor[B, T_text]]       # token ids; empty when user silent

# Outputs per tick
CursorChunk   : Tensor[B, 6, 2]                   # 6 (dx, dy) samples flow-matched
EventTokens   : Tensor[B, K_evt]                  # discrete; K_evt small (e.g. 4)
MonoTokens    : Tensor[B, K_mono]                 # discrete; mostly empty
DispatchTokens: Tensor[B, K_disp]                 # discrete; rare
ScreenPred    : Tensor[B, N_patch, D_patch_out]   # next-frame patch tokens (train-only at infer)
```

## 2. Embeddings (encoder-free early fusion)

```python
class HBCFusion(nn.Module):
    def __init__(self, d_model):
        self.hmlp_patch = HMLPEmbed(patch_size=40, d_out=d_model)         # TML hMLP
        self.dmel_audio = DMelEmbed(d_out=d_model)                         # TML dMel
        self.cursor_in  = nn.Linear(3, d_model)                            # (x,y,btn) -> d
        self.event_tok  = nn.Embedding(N_EVENT_VOCAB, d_model)
        self.text_tok   = nn.Embedding(N_TEXT_VOCAB, d_model)
        self.dom_tok    = nn.Embedding(N_DOM_VOCAB, d_model)
        self.modality   = nn.Embedding(N_MODALITY, d_model)                # signals stream-id
        self.tick_pos   = SinusoidalPos(d_model)

    def forward(self, t_idx, screen, cursor, dom, text, audio=None):
        # All embeds + add modality + tick position
        screen_e = self.hmlp_patch(screen) + self.modality(MOD_SCREEN) + self.tick_pos(t_idx)
        cursor_e = self.cursor_in(cursor) + self.modality(MOD_CURSOR) + self.tick_pos(t_idx)
        dom_e    = self.dom_tok(dom) if dom is not None else None
        text_e   = self.text_tok(text) if text is not None else None
        audio_e  = self.dmel_audio(audio) if audio is not None else None
        return concat_nonempty([screen_e, cursor_e, dom_e, text_e, audio_e])  # [B, T_in_tick, d]
```

## 3. Backbone — RQ-Transformer (Moshi-style)

```python
class HBCBackbone(nn.Module):
    def __init__(self, d_model=4096, n_temp_layers=32, n_depth_layers=6, K_channels=5):
        # K_channels = {cursor, event, mono, dispatch, screen_pred}
        self.temporal = CausalTransformer(d_model, n_temp_layers)          # ~7B
        self.depth    = CausalTransformer(d_model, n_depth_layers)         # ~200M
        self.K = K_channels
        self.channel_emb = nn.Embedding(K_channels, d_model)

    def forward_temporal(self, x_in_tick, kv_cache):
        # x_in_tick: [B, T_in_tick, d]. KV cache persists across ticks (streaming session).
        h, kv_cache = self.temporal(x_in_tick, kv_cache=kv_cache, causal=True)
        # h[-1] is the temporal-trunk hidden at this tick's "now"
        return h, kv_cache

    def forward_depth(self, ctx_now):
        # ctx_now: [B, d] (last hidden from temporal). Depth trunk runs K times, one per output channel.
        seq = []
        for k in range(self.K):
            ch = ctx_now + self.channel_emb(k)
            ch_out, _ = self.depth(ch.unsqueeze(1))   # tiny seq of length k+1 internally
            seq.append(ch_out.squeeze(1))
        # seq: list of K channel-conditioned hiddens
        return seq  # [K] x [B, d]
```

## 4. Heads

```python
class HBCHeads(nn.Module):
    def __init__(self, d_model):
        self.cursor_flow = FlowMatchingHead(d_model, out_dim=2, chunk=6)   # 6 steps of (dx,dy)
        self.event_head  = nn.Linear(d_model, N_EVENT_VOCAB)
        self.mono_head   = nn.Linear(d_model, N_TEXT_VOCAB)
        self.disp_head   = nn.Linear(d_model, N_DISPATCH_VOCAB)
        self.screen_head = ScreenAR_FlowMatch(d_model, n_patch=256, d_patch=D_patch_out)  # Dreamer-4 shortcut-forcing

    def forward(self, channel_hiddens, train_world_head=True):
        h_cur, h_evt, h_mono, h_disp, h_scr = channel_hiddens
        out = {
            "cursor": self.cursor_flow.sample(h_cur),                       # flow-matched trajectory
            "event":  self.event_head(h_evt),                                # logits over event vocab
            "mono":   self.mono_head(h_mono),                                # logits over text vocab
            "disp":   self.disp_head(h_disp),                                # logits over dispatch vocab
        }
        if train_world_head:
            out["screen"] = self.screen_head.sample(h_scr)                  # next-frame patch prediction
        return out
```

## 5. The tick loop (inference)

```python
class HBC(nn.Module):
    def __init__(self, ...):
        self.fuse = HBCFusion(d_model)
        self.backbone = HBCBackbone(d_model)
        self.heads = HBCHeads(d_model)
        self.kv_cache = init_kv_cache()
        self.dispatch_inbox = AsyncQueue()    # background reasoner streams plan tokens here

    @torch.inference_mode()
    def tick(self, screen, cursor, dom, text, audio):
        # 1. Drain any plan tokens from the background reasoner inbox
        plan_tokens = self.dispatch_inbox.drain_nonblocking()  # may be empty

        # 2. Build this tick's input
        x_in = self.fuse(self.t_idx, screen, cursor, dom, text, audio)
        if plan_tokens:
            x_in = torch.cat([x_in, plan_tokens], dim=1)

        # 3. Temporal trunk forward
        h, self.kv_cache = self.backbone.forward_temporal(x_in, self.kv_cache)
        ctx_now = h[:, -1, :]

        # 4. Depth trunk per channel
        chans = self.backbone.forward_depth(ctx_now)

        # 5. Head sampling (skip world head at inference for budget)
        out = self.heads(chans, train_world_head=False)

        # 6. Dispatch: if dispatch head emits a dispatch token, fire async to bg reasoner
        if out["disp"].argmax(-1).item() != DISP_NONE:
            self.fire_dispatch(out["disp"], context=h)   # non-blocking

        # 7. Sliding window + episodic compression (drop / compress KV older than W ticks)
        self.kv_cache = compress_kv(self.kv_cache, window=300, summarizer=self.episodic_summarizer)

        self.t_idx += 1
        return out  # action chunk to execute this tick
```

## 6. Training loss (joint world-model + policy)

```python
def hbc_loss(model, batch):
    """
    batch: B trajectories, each with synchronized streams and action labels.
    For each tick t in trajectory, compute:
      - L_screen[t]  = AR/flow-match loss on next-tick screen-patch tokens
      - L_cursor[t]  = flow-matching loss on next-tick cursor chunk
      - L_event[t]   = CE on next-tick event tokens
      - L_mono[t]    = CE on next-tick monologue tokens (if present)
      - L_dispatch[t]= CE on next-tick dispatch tokens (if present)
    """
    total = 0.0
    out = model.forward_train(batch)  # teacher-forced through whole trajectory
    for t in range(T):
        L_screen   = flow_matching_loss(out["screen"][t], batch.screen[t+1])      # weight a
        L_cursor   = flow_matching_loss(out["cursor"][t], batch.cursor[t+1])      # weight b
        L_event    = F.cross_entropy(out["event"][t], batch.event[t+1])           # weight c
        L_mono     = F.cross_entropy(out["mono"][t], batch.mono[t+1], mask=...)   # weight d
        L_dispatch = F.cross_entropy(out["disp"][t], batch.disp[t+1], mask=...)   # weight e
        total += a*L_screen + b*L_cursor + c*L_event + d*L_mono + e*L_dispatch
    return total / T

# Stage 0 weights: a=1.0, b=0.3, c=0.3, d=0.5, e=0.5   (action labels are IDM-noisy)
# Stage 1 weights: a=0.3, b=1.0, c=1.0, d=1.0, e=1.0   (action labels are gold)
```

## 7. Dispatch protocol

```python
# Foreground side
def fire_dispatch(self, disp_logits, context):
    payload = pack_dispatch(
        task_summary=self.episodic_summarizer(context),
        proposed_action=self.last_action,
        screen_hash=hash(self.last_screen),
        timestamp=time.time(),
    )
    asyncio.create_task(self.bg_reasoner.ask(payload, callback=self.dispatch_inbox.push))

# Background reasoner (separate process / API call)
def bg_ask(payload):
    plan = call_frontier_model(payload, system="you advise a foreground GUI agent")  # blocking inside this proc
    plan_tokens = tokenize(plan)
    return plan_tokens

# Foreground reads inbox each tick (non-blocking). If empty, carry on with last plan.
```

## 8. Dream-RL rollout (Stage 2)

```python
def dream_rollout(model, task_desc, init_screen, K_horizons=64):
    # Teacher-force the model on init_screen + task desc, then let it autoregress
    # both screen-prediction and action-prediction for K_horizons ticks.
    # Capture monologue head's predicted task-completion confidence at each tick.
    model.eval()
    state = encode_init(task_desc, init_screen)
    horizon = []
    for t in range(K_horizons):
        out = model.tick_dream(state)   # uses model.screen_head autoregressively, no real frame
        state = out
        horizon.append(out)
        if out["mono_conf"] > 0.95: break
    return horizon

def dream_rl_step(model, executor, task_desc, init_screen):
    # Generate N candidate dreams
    dreams = [dream_rollout(model, task_desc, init_screen) for _ in range(N)]
    # Score each by mono_conf at terminal tick
    best = max(dreams, key=lambda d: d[-1]["mono_conf"])
    # Replay best's action sequence in real executor
    real_traj = executor.replay(task_desc, init_screen, [d["cursor"] + d["event"] for d in best])
    real_reward = executor.score(real_traj)
    # Advantage = real_reward - predicted_reward
    adv = real_reward - best[-1]["mono_conf"]
    return policy_gradient_loss(best, adv) + world_correction_loss(best, real_traj)
```

## 9. Open gaps in this sketch

These pieces are placeholders and need real engineering:

- `FlowMatchingHead` for cursor: pick a concrete formulation (rectified flow vs Dreamer-4 shortcut-forcing). Trade-offs in COMPUTE.md unresolved.
- `ScreenAR_FlowMatch` for next-frame: at 256 patches/tick this is the heaviest head. Reuse Dreamer-4's design directly.
- `episodic_summarizer`: compress old KV into ~64 tokens. Could be the trunk itself attending over its own old KV; or StreamAgent's hierarchical structure.
- `compress_kv`: when to compress; how much to keep verbatim. Sliding window of 300 ticks (30 s) + episodic summary beyond — needs measurement, not theory.
- Background reasoner channel format: how does plan come back? Plain text tokens? Action-suggestion tokens that share the dispatch vocab? Probably the latter for training-time joint optimization.
- Wake-word / VAD on audio input — out of scope for v0; assume push-to-talk.

## 10. What this pseudocode commits us to

- One model, one forward pass per tick, multi-head output.
- KV cache that persists across ticks (streaming session).
- Plan stream from background reasoner re-enters the trunk as ordinary tokens — uniform interface.
- World-model head is optional at inference but always trained.
- All loss terms applied teacher-forced; no scheduled-sampling complexity in v0.
- Continuous cursor head and discrete event head coexist in the same depth-trunk loop — no separate motor network.
