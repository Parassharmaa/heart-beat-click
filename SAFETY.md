# Heart Beat Click — Safety Analysis

Date: 2026-05-14. Grounded in the existing CUA-vulnerability literature, not speculation. Reference: *A Systematization of Security Vulnerabilities in Computer Use Agents* (arxiv 2507.05445).

## 1. What duplex changes about the threat model

Both directions matter.

### Worse with duplex (faster bad actions)
- Closed-loop irreversibility: half-duplex agents act every 1–5 s; the user (or a watchdog) has time to interrupt. HBC at 100 ms tick can complete a destructive sequence (e.g., click → confirm → submit) in under 500 ms — below human reaction time.
- Continuous cursor motion can complete drag-and-drop deletions before any preview frame is rendered.
- Audio side-channel: if HBC accepts streaming voice input, that input itself can be a prompt-injection vector (acoustic adversarial → voice command).

### Better with duplex (continuous perception loop)
- The CUA-vuln paper explicitly identifies as a weakness: *"no continuous perception loop with actions typically executed after a static planning stage."* HBC's joint world+policy loss directly closes this: at every tick the model predicts what the screen should look like and can detect divergence (modal popups, unexpected confirmation dialogs, page redirects) within one tick.
- Inner-monologue stream is observable in real time → human supervisor can read the model's stated intent before any action executes.
- Dispatch protocol lets the foreground escalate any high-confidence-but-novel action to the background reasoner for second-opinion approval before motor commit.

## 2. Required mitigations for v0 release

### Revocable-action gating
- Classify every action token as: trivial (mouse hover, scroll), reversible (click, type into text field that hasn't been submitted), irreversible (form submit, "send", "delete", payment, file-system write outside scratch).
- Trivial: always allowed.
- Reversible: allowed; logged.
- Irreversible: require explicit confirmation from a separate verifier model or human-in-the-loop, with a configurable bypass for tasks the user has explicitly pre-authorized.
- This is independent of the duplex architecture and matches Anthropic's computer-use guidance.

### Confirmation watchdog (parallel head)
- A small (sub-B parameter) classifier head reading the same trunk hidden state predicts P(this action is irreversible). When P > threshold and the foreground is about to emit the action token, insert a one-tick pause and ping the background reasoner with the proposed action + current frame.
- Foreground does not block on the response; it emits the action only if approval token arrives within N ticks. Otherwise it emits a "wait" action and shows the user.

### Prompt-injection containment
- The trunk receives only the *visual* page content; it does not interpret embedded "ignore previous instructions" text as system instructions (we train a system-prompt-grounded loss in S1 to make the inner-monologue stay aligned with the explicit task description rather than page-content-suggested goals).
- Network egress from the browser VM is on an allowlist controlled by the user's task, not the agent.
- Cross-origin form submissions to non-allowlisted domains are blocked at the VM level.
- Acoustic input is gated by a wake-word + voiceprint match by default.

### Surveillance / privacy
- HBC requires capturing screencast. For Stage 0 (public corpora), this is already public.
- For Stage 1 contractor data, all capture happens inside disposable VMs containing only synthetic personas; no real user data is recorded.
- For deployment, screencast capture is opt-in per-app and shows a persistent on-screen indicator.
- All recorded screencast is filtered by an OCR-based PII redactor before storage; raw frames are not retained beyond a 24h training window.

### Auditability
- Every action emitted by HBC is logged with: timestamp, frame hash, inner-monologue snippet at that tick, dispatch interactions, model checkpoint id, system-prompt task description.
- Logs are signed and append-only. Replay-from-log is the canonical post-hoc audit interface.

## 3. Open safety questions for the paper

- Does the joint world-model loss create a *more* hackable agent (by giving prompt-injection content a way to nudge the predicted next-frame distribution, which then biases the action distribution)? Empirical test: red-team with adversarial DOM payloads, compare to half-duplex baseline.
- Does flow-matched cursor make trajectory-based attacks (e.g., draw a malicious signature) easier? Test: target a phishing-signature task; can HBC be coaxed to sign in the wrong UI?
- Does the background reasoner protocol leak the user's task context to a third-party model provider? Solution: dispatch payload is summarized + redacted before egress; user controls which background provider is allowed.

## 4. Pre-release red-team protocol

Before any open-source release of Stage 1 checkpoints:
1. Adversarial DOM injection tests on a curated 50-task suite.
2. Acoustic prompt-injection tests (10 samples).
3. Cross-origin egress attempts (20 scenarios).
4. Irreversibility-classifier false-negative rate measured on a curated 200-task set.
5. Compare HBC's mid-action correction rate to baseline computer-use agents — does duplex actually let the user correct, in practice?
6. External red-team review (one independent security team) — required pre-release.

Withhold Stage 2 (dream-RL) checkpoint until 5 and 6 pass.

## 5. Statement for the paper

> "Heart Beat Click expands both sides of the safety surface for computer-use agents. The duplex tick collapses the human-intervention window for irreversible actions from seconds to fractions of a second, requiring on-line mitigations (revocable-action gating, confirmation watchdog, allowlisted egress) that we describe in detail. Conversely, the joint world-and-policy objective explicitly closes the 'no continuous perception loop' gap identified by prior CUA-vulnerability work, by training the model to verify expected-vs-actual screen state on every tick. We release HBC-Bench and Stage 0/Stage 1 checkpoints with full safety documentation. Stage 2 dream-RL checkpoints will be released only after independent red-team review."
