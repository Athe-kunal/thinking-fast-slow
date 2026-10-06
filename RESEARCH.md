# Research log: thinking fast and slow with one tri-mode model

Running record of what we test, the assumptions behind each test, and what we
found. Newest experiment last within "Experiments". Every number here comes
from a run artifact under `runs/` (git-ignored, local to the machine) or from a
cited source; anything not verified is marked as such.

**How experiments are run (from E16 on):** each experiment is a YAML file in `configs/experiments/` run by `scripts/run_experiments.py` (outputs in `runs/experiments/<run>/`, resolved config saved with results); `scripts/summarize_experiments.py` tabulates results with paired tests. E13-E15 were launched by hand before the runner existed; their configs are recorded in `configs/experiments/e13_*`, `e14_*`, `e15_*`.

**Maintenance rule:** every new experiment gets an entry (question, setup,
assumptions, results, conclusion, artifacts) and any new global assumption goes
into the assumption register.

## 1. Idea

Use `nvidia/Nemotron-Labs-Diffusion-3B` as both a fast thinker (block
diffusion: many tokens per forward pass) and a slow thinker (autoregressive,
one token per pass), with a **router** that decides, from the model's causal
hidden state, which mode decodes the next segment. Diffusion can commit both
"fast thinking" tokens and action tokens; AR handles deliberate reasoning and
can validate diffusion output cheaply. Long term: train the router (RL on task
outcome + compute cost, possibly with a supervised warm start), backbone frozen.

## 2. Setup

| item | value |
|---|---|
| Model | `nvidia/Nemotron-Labs-Diffusion-3B` (3.83B params, bf16), Ministral3 backbone |
| Hardware | A100 80GB; **allowed GPUs: 3 (and 2 since 2026-10-03)**; shared machine |
| Driver | 550 (CUDA 12.4); main env torch 2.6.0+cu124, transformers 5.x |
| Model code | Local editable copy in `src/nemotron/` (HF remote code, unmodified) |
| Engine | `src/engine.py` modes `ar`, `dlm`, `mix`; `src/interleave.py` (routed decoding); `src/router.py` (routers) |
| Serving | `src/server.py` (OpenAI-compatible, mode via model name); SGLang fork in `.venv-sglang` (Sec. 5, E3) |

## 3. Verified facts about the architecture

- **One backbone, two attention patterns.** A single `encoder`
  (`Ministral3Model`) plus one LM head (`diffusion_head`). Mode = per-layer
  flag `self_attn.diffusion_lm` (True: bidirectional, False: causal).
- **The KV cache only ever holds causal state.** Diffusion denoising passes run
  with `use_cache=False`; a finished block is committed with one causal pass.
- **Switching on the causal hidden state is exact** (E2): at every switch point
  the router's hidden state matches a fresh causal pass over the same tokens
  (cosine 0.9998-1.0, same next token), and a committed diffusion block can be
  cut at any position k and continued in AR exactly.
- `ar_generate` / `generate` flip the flag in place; alternating them is safe
  (both set it before use).
- Fixed routers reproduce stock decoders token for token: `FixedRouter("ar")`
  == `ar_generate`, `FixedRouter("dlm")` == `generate(block 32, thr 0.9)`.

## 4. Assumption register (applies unless an experiment says otherwise)

| # | assumption | why | risk / how it could mislead |
|---|---|---|---|
| A1 | Greedy decoding everywhere (temperature 0) | Deterministic, matches HF examples | One sample per item; no sampling variance measured |
| A2 | Diffusion block 32, confidence threshold 0.9, first token seeded by the causal pass ("causal context") | HF chat-example defaults | Paper's diffusion setting uses block **8**; block size likely matters for code |
| A3 | Router decides per **segment**; in E4/E5 AR segments = 32 tokens so p = expected diffusion token share | Interpretable p | Coarse: switches only every 32 tokens; not how a real router would work (AR can hand over after any token) |
| A4 | Random router: one seed per (setting, benchmark, item) via sha256 | Resumable, worker-independent | Only one router draw per item; CIs cover item variance, not router-seed variance |
| A5 | Efficiency metric = tokens per forward pass (prefill = 1 pass, each denoising step = 1, each block commit = 1) | Hardware-independent | Treats a 32-token pass as costing the same as a 1-token pass; not wall-clock. Close to, not identical with, NVIDIA's TPF counting |
| A6 | Chat template defaults: `enable_thinking=False` (thinking off) | Template default; NVIDIA `evaluate.py` doesn't set it | Paper's numbers may use thinking (unknown) |
| A7 | Long prompts prefilled in 8192-token chunks, logits only at the last position | Memory (fix for OOM at 66k-token contexts) | Verified identical outputs on a 17k-token prompt |
| A8 | Answer budget 1024 tokens | Throughput | Paper uses 8192; only 0-2 truncations per setting in E5 |
| A9 | tok/s numbers are **not** speed measurements when several workers share a GPU | - | Use tokens/forward for efficiency |

## 5. Experiments

### E1. Harbor toy agent tasks (2026-10-02)

- **Question:** does the full agent pipeline (server + mini-swe-agent) run in AR and diffusion modes?
- **Setup:** 2 self-written Harbor-format tasks (`tasks/hello-world`, `tasks/sum-numbers`), mini-swe-agent text-based config, step limit 15, hub remote-code model, diffusion block 32 / thr 0.9.
- **Results:**

  | task | mode | reward | exit | steps |
  |---|---|---|---|---|
  | hello-world | ar | 1 | Submitted | 2 |
  | hello-world | dlm | 1 | LimitsExceeded (never submitted) | 15 |
  | sum-numbers | ar | 1 | Submitted | 2 |
  | sum-numbers | dlm | 0 | LimitsExceeded (looped on relative path `app/sum.py`) | 15 |
- **Conclusion:** pipeline works; n=2, no statistical weight.
- **Artifacts:** `runs/*.traj.json`.

### E2. Can we switch modes from the last causal hidden state? (2026-10-03)

- **Setup:** `scripts/check_switch.py`, GPU 3, CycleRouter (ar8 dlm32 ...), 96 tokens; mid-block cut at k=11 then 16 AR tokens.
- **Results:** switch-point cosine vs fresh causal pass 0.99982-1.00000, same next token at all 6 switch points; mid-block positions 3/11/31 cosine 0.99976-0.99988; crop-then-AR == `ar_generate` from same prefix: **True**. Differences are bf16 incremental-vs-full-prefill noise.
- **Conclusion:** routing on the causal hidden state is exact; diffusion blocks can be cut at any position after the causal commit; no switching mid-denoising.

### E3. SGLang DLLM fork veracity (2026-10-03)

- **Setup:** fork `hutm/sglang` @ `23472a0` (branch `upstream/2-dllm-linearspec`; NVIDIA's guide branch `upstream/2-dllm-lora-ar` no longer exists); upstream PR sgl-project/sglang#25803 open, changes requested. CUDA 13 pins swapped for CUDA 12.x builds (`configs/sglang/overrides.txt`). `scripts/compare_sglang.py`: 30 GSM8K questions, own prompt with `####`, 512 tokens, greedy, concurrency 1, GPU 3.
- **Results:**

  | mode | SGLang acc | HF acc | identical outputs | SGLang tok/s | HF tok/s |
  |---|---|---|---|---|---|
  | linear_spec | 86.7% | 86.7% | 19/30 | 450 | 187 |
  | dlm (FastDiffuser, thr 0.9) | 83.3% | 86.7% | 8/30 | 291 | 116 |
  | ar (`ar_mode` + FastDiffuser) | **36.7%** | 86.7% | 0/30 | 109 | 51 |
- **Conclusion:** SGLang LinearSpec is trustworthy and ~2.4x faster; SGLang "AR mode" is broken (diffusion loop over causal attention, only tested for non-empty output upstream). One decoding mode per server; no router support.

### E4. SWE-bench Verified with a random router (2026-10-03, stopped early)

- **Question:** how does an untrained (random) router do on real SWE tasks at p(diffusion) = 0.2 vs 0.8?
- **Setup:** `scripts/swebench_router_experiment.py`; 50 instances sampled from Verified (seed 0; `runs/swebench_random_router/instances.json`); mini-swe-agent `swebench_backticks.yaml` (text actions); step limit 50, 1024 tokens/step, 60 s per command, no context truncation; `mix` mode, RandomRouter, AR segments 32; 2 lanes x 2 settings = 4 servers on GPU 3; scoring with `swebench` 4.1.0 harness (validated: gold patch resolves, test-only patch does not).
- **Results (stopped after ~19 runs per setting):**

  | | p=0.2 | p=0.8 |
  |---|---|---|
  | finished / scored | 18 / 10 | 19 / 10 |
  | resolved | 0 | 0 |
  | non-empty patch | 0/18 | 0/19 |
  | exit statuses | LimitsExceeded 15, Submitted (empty) 3 | LimitsExceeded 15, Submitted (empty) 2, RepeatedFormatError 2 |
  | mean steps | 43.6 | 45.3 |
  | format errors / step | 0.03 | 0.06 |
  | diffusion token share | 18.9% | 78.9% |
  | tokens / forward | 1.16 | 2.30 |
- **Trace analysis (18 runs each):** opened the file the gold fix changes 0/18 vs 2/18; ran tests 2/18 vs 2/18; repeated one command 5+ times 16/18 vs 16/18. Typical failures: blind repo-wide `sed` with no reading, find/sed loops, copying the prompt's placeholder `git diff -- path/to/file1 ...`.
- **Conclusion:** the 3B model is at a floor of zero on SWE-bench Verified in both settings, so this task carries **no signal about routing**. Pure AR (p=0) was not run.
- **Issues found and fixed during E4:** (1) prefill computed full-prompt logits (~10 GB at 40k tokens) -> last-position only; (2) OOM at 66k-token contexts -> chunked prefill (A7) + per-server memory cap; (3) harness summary file missing after cleanup crash -> read per-instance `report.json`. Run restarted from scratch after each code change.
- **Artifacts:** `runs/swebench_random_router/` (`summary.json`, per-lane trajectories and request logs).

### E5. Random router on GSM8K and HumanEval (2026-10-03)

- **Question:** does random AR/diffusion routing keep accuracy, and how does it trade accuracy for efficiency?
- **Setup:** `scripts/eval_router_benchmarks.py`; full GSM8K test (1319) and HumanEval (164); p in {0, 0.2, 0.8, 1}; segments 32/32 (A3); 1024 tokens; GSM8K prompt and scoring copied from NVIDIA `evaluate.py`; HumanEval: own prompt (full function in one code block), first code block, stdlib-only network-less Docker sandbox, 10 s timeout. 12 workers on GPUs 2 and 3 (restart from 6 workers on GPU 3 kept finished records).
- **Results:**

  | benchmark | p | accuracy | 95% CI | diffusion share | tokens/forward | mean length |
  |---|---|---|---|---|---|---|
  | GSM8K | 0 (AR) | 87.3% | 85.4-89.0 | 0% | 1.00 | 177 |
  | GSM8K | 0.2 | 87.5% | 85.6-89.2 | 19.8% | 1.14 | 174 |
  | GSM8K | 0.8 | 87.8% | 85.9-89.5 | 79.8% | 2.01 | 174 |
  | GSM8K | 1 (diffusion) | 87.3% | 85.4-89.0 | 100% | 2.72 | 173 |
  | HumanEval | 0 (AR) | 80.5% | 73.8-85.8 | 0% | 1.00 | 210 |
  | HumanEval | 0.2 | 79.3% | 72.4-84.8 | 19.7% | 1.16 | 273 |
  | HumanEval | 0.8 | 69.5% | 62.1-76.0 | 79.7% | 1.96 | 374 |
  | HumanEval | 1 (diffusion) | 72.6% | 65.3-78.8 | 100% | 2.46 | 415 |

  Paired exact McNemar tests (items solved by only one setting):
  - HumanEval: AR vs 0.8: 21 vs 3, **p<0.001**; AR vs 1.0: 20 vs 7, **p=0.019**; AR vs 0.2: 8 vs 6, p=0.79; 0.8 vs 1.0: 4 vs 9, p=0.27.
  - GSM8K: all pairs p >= 0.44.
- **Checks:** HumanEval drop is not an extraction artifact (at most 1/164 outputs per setting have more than one code block; inspected failures are genuinely wrong code). Longer diffusion outputs come from added explanation text. Our AR/diffusion GSM8K (87.3%) matches NVIDIA's 3B table (87.87% AR, 88.40% Diff.).
- **Conclusion:** switching never breaks generation. On math, routing is free (2-2.7x fewer forward passes, no accuracy change). On code, diffusion costs ~8-11 points, so a learned router has real headroom there: target AR's ~80% at diffusion-level efficiency.
- **Caveats:** A2 (block 32 vs paper's 8), A4 (one router draw), own HumanEval prompt (absolute numbers not comparable to NeMo-Skills).
- **Artifacts:** `runs/router_bench/` (`summary.json`, `scored.jsonl`, `worker*.jsonl`).

### E6. Entropy router (0.5 nats) on GSM8K and HumanEval (2026-10-03)

- **Question:** does routing on next-token entropy beat random routing at a similar diffusion share?
- **Setup:** identical to E5 (same prompts, scoring, block 32 / thr 0.9, 32-token AR segments, 1024 tokens, greedy), router `entropy:0.5`: at each segment boundary, compute the entropy (nats, float32, full vocabulary) of the causal next-token distribution; diffusion if < 0.5, else AR. Boundaries always follow a causal pass (AR token or block commit). 12 workers on GPUs 2 and 3; records written next to E5's in `runs/router_bench/`. Deterministic router (no seed).
- **Results:**

  | benchmark | router | accuracy | 95% CI | diffusion share | tokens/forward | mean length |
  |---|---|---|---|---|---|---|
  | GSM8K | entropy:0.5 | 87.7% | 85.8-89.4 | 89.8% | 2.34 | 173 |
  | GSM8K | random:0.8 (E5) | 87.8% | 85.9-89.5 | 79.8% | 2.01 | 174 |
  | GSM8K | random:1.0 (E5) | 87.3% | 85.4-89.0 | 100% | 2.72 | 173 |
  | HumanEval | entropy:0.5 | **76.2%** | 69.2-82.1 | 87.5% | 2.17 | 420 |
  | HumanEval | random:0.0 = AR (E5) | 80.5% | 73.8-85.8 | 0% | 1.00 | 210 |
  | HumanEval | random:0.8 (E5) | 69.5% | 62.1-76.0 | 79.7% | 1.96 | 374 |
  | HumanEval | random:1.0 (E5) | 72.6% | 65.3-78.8 | 100% | 2.46 | 415 |

  Paired exact McNemar (only-A vs only-B solved):
  - HumanEval: entropy vs AR 11 vs 18, p=0.27; entropy vs random:0.8 16 vs 5, **p=0.027**; entropy vs random:1.0 10 vs 4, p=0.18.
  - GSM8K: entropy vs AR p=0.73; vs random:0.8 p=1.0; vs random:1.0 p=0.45.
- **Conclusion (revised after E7):** on HumanEval the entropy router beat random:0.8 (+6.7 points, p=0.027) while using more diffusion, but E7 shows random:0.8 was an unusually low draw (random:0.9 scored 73.8%). Against the matched-share random:0.9 the gain is +2.4 points, not significant (p=0.45). Not distinguishable from pure AR (-4.3, p=0.27) or pure diffusion (+3.6, p=0.18) at n=164. GSM8K: no differences, as in E5.
- **Caveats:** 0.5 nats picks diffusion ~88-90% of the time, so this is one point on the curve, not a matched-share comparison (no random run at ~88%); random routers have one draw per item (A4); E5/E6 block-32 setting (A2); outputs stay long (420 tokens) like diffusion.
- **Artifacts:** `runs/router_bench/` (`setting: "entropy:0.5"` records; `summary.json`, `scored.jsonl`).

### E7. Entropy threshold sweep vs random routing on HumanEval (2026-10-03)

- **Question:** what accuracy/efficiency curve do entropy routers trace, and do they beat random routing at matched diffusion share?
- **Setup:** identical to E5/E6 (HumanEval only); new settings entropy:{0.05, 0.1, 0.25, 1.0, 2.0} and random:{0.5, 0.9}; 12 workers on GPUs 2 and 3.
- **Results (HumanEval, n=164):**

  | router | accuracy | diffusion share | tokens/forward | mean length |
  |---|---|---|---|---|
  | random:0.0 (AR) | 80.5% | 0% | 1.00 | 210 |
  | random:0.2 | 79.3% | 19.7% | 1.16 | 273 |
  | random:0.5 | 76.2% | 47.2% | 1.44 | 317 |
  | entropy:0.05 | 75.6% | 74.3% | 1.91 | 419 |
  | entropy:0.1 | 75.0% | 78.3% | 1.98 | 422 |
  | random:0.8 | 69.5% | 79.7% | 1.96 | 374 |
  | entropy:0.25 | 75.6% | 82.6% | 2.07 | 422 |
  | entropy:0.5 | 76.2% | 87.5% | 2.17 | 420 |
  | random:0.9 | 73.8% | 90.6% | 2.22 | 395 |
  | entropy:1.0 | 75.6% | 97.0% | 2.38 | 414 |
  | entropy:2.0 | 72.6% | 99.9% | 2.45 | 416 |
  | random:1.0 (diffusion) | 72.6% | 100% | 2.46 | 415 |

  Matched-share paired McNemar: entropy:0.5 vs random:0.9 10 vs 6, p=0.45; entropy:1.0 vs random:1.0 6 vs 1, p=0.13; entropy:0.1 vs random:0.8 16 vs 7, p=0.09; random:0.9 vs random:0.8 11 vs 4, p=0.12. Mean accuracy entropy 0.05-1.0 = 75.6% vs random 0.8-1.0 = 72.0%.
- **Findings:**
  1. **Entropy at segment boundaries is almost always tiny.** Even a 0.05-nat threshold sends 74% of tokens to diffusion, so this router cannot reach the low-diffusion part of the curve where accuracy is near AR's.
  2. **Flat plateau at ~75-76%** for thresholds 0.05-1.0 (74-97% diffusion), then a drop to pure-diffusion level at 2.0. The gap to AR (~4-5 points) does not close with any threshold.
  3. **Entropy is consistently a little above random at similar share** (+2.4 to +3.6 points at the matched points, +3.6 on average), and every entropy point at >= 74% share sits above the random points at 80-100%, but no single matched comparison is significant at n=164.
  4. **Random routing is noisy** (random:0.8 69.5% < random:0.9 73.8%): one router draw per item (A4) can move results by ~4 points, which inflated E6's significance.
  5. Outputs stay long (~420 tokens) whenever diffusion share is high, even with the entropy router.
- **Conclusion:** next-token entropy at segment boundaries is a weak but plausibly real signal (small, consistent edge over random) with too little dynamic range to keep code accuracy near AR. A better signal or finer granularity is needed (entropy over a window or a diffusion draft; AR decisions every 1-8 tokens; smaller diffusion blocks).
- **Artifacts:** `runs/router_bench/` (`summary.json`, `scored.jsonl`).

### E8. Per-token entropy check while in AR (2026-10-04)

- **Question:** if the router checks entropy after *every* AR token and switches to diffusion as soon as it drops below the threshold, does it keep AR's accuracy at higher efficiency?
- **Setup:** as E5-E7 but AR segments of 1 token (setting suffix `@ar1`): after each AR token, next-token entropy < threshold -> one 32-token diffusion block, else one more AR token. Diffusion blocks unchanged (32 tokens, committed whole; the next check happens after the block's causal commit). Settings: entropy:0.5@ar1 on GSM8K + HumanEval, entropy:0.05@ar1 on HumanEval. 12 workers on GPUs 2 and 3.
- **Results:**

  | benchmark | router | accuracy | 95% CI | diffusion share | tokens/forward |
  |---|---|---|---|---|---|
  | GSM8K | entropy:0.5@ar1 | 87.6% | 85.8-89.3 | 99.5% | 2.68 |
  | HumanEval | entropy:0.5@ar1 | 73.8% | 66.6-79.9 | 99.4% | 2.44 |
  | HumanEval | entropy:0.05@ar1 | 73.8% | 66.6-79.9 | 98.3% | 2.46 |
  | HumanEval | entropy:0.5 (32-token AR, E6) | 76.2% | 69.2-82.1 | 87.5% | 2.17 |
  | HumanEval | pure diffusion (E5) | 72.6% | 65.3-78.8 | 100% | 2.46 |
  | HumanEval | pure AR (E5) | 80.5% | 73.8-85.8 | 0% | 1.00 |

  AR usage (HumanEval): 0.5 nats -> 1.7 AR runs per answer, mean run 1.4 tokens (max 6); 0.05 nats -> 3.7 runs, mean 1.9 tokens (max 12).
  Paired McNemar: entropy:0.5@ar1 vs pure diffusion 5 vs 3, p=0.73; vs AR 9 vs 20, p=0.061; vs entropy:0.5 (32-token AR) 6 vs 10, p=0.45. GSM8K vs pure diffusion p=0.49.
- **Findings:**
  1. **Per-token checking makes the router almost pure diffusion.** Entropy rises above the threshold for one token, AR decodes that token, entropy falls, and diffusion takes the next 32-token block. Even 0.05 nats leaves AR only ~2% of tokens.
  2. **Handing AR the single uncertain tokens at block boundaries does not help:** accuracy equals pure diffusion (73.8% vs 72.6%, p=0.73). The efficiency equals pure diffusion too.
  3. **The asymmetry is the problem:** the router can leave AR after any token, but once in diffusion it commits 32 tokens with no check, and the uncertain tokens inside a block are decided by diffusion itself. Boundary entropy (right after a commit) is almost always low.
- **Conclusion:** the decisions that matter are inside diffusion blocks. Next idea: after a block's causal commit pass (which already yields AR logits for every block position, E2), cut the block at the first position whose causal entropy (or AR disagreement) is high and continue in AR from there; or use smaller diffusion blocks (8).
- **Artifacts:** `runs/router_bench/` (`setting` ends in `@ar1`).

### E9. Entropy router with 8-token AR segments (2026-10-04)

- **Question:** between E6 (AR commits 32 tokens per decision) and E8 (1 token), does committing 8 AR tokens per entropy decision do better?
- **Setup:** as E6/E8 with setting `entropy:0.5@ar8`: when next-token entropy >= 0.5 nats, decode 8 AR tokens, then check again; below 0.5, one 32-token diffusion block. GSM8K + HumanEval, 12 workers on GPUs 2 and 3.
- **Results (all entropy routers at 0.5 nats, by AR segment length):**

  | benchmark | AR segment | accuracy | 95% CI | diffusion share | tokens/forward |
  |---|---|---|---|---|---|
  | HumanEval | 32 (E6) | 76.2% | 69.2-82.1 | 87.5% | 2.17 |
  | HumanEval | **8** | 73.8% | 66.6-79.9 | 96.0% | 2.38 |
  | HumanEval | 1 (E8) | 73.8% | 66.6-79.9 | 99.4% | 2.44 |
  | HumanEval | pure diffusion | 72.6% | 65.3-78.8 | 100% | 2.46 |
  | HumanEval | pure AR | 80.5% | 73.8-85.8 | 0% | 1.00 |
  | GSM8K | 32 (E6) | 87.7% | 85.8-89.4 | 89.8% | 2.34 |
  | GSM8K | **8** | 87.9% | 86.1-89.6 | 96.8% | 2.60 |
  | GSM8K | 1 (E8) | 87.6% | 85.8-89.3 | 99.5% | 2.68 |

  AR usage with 8-token segments (HumanEval): 1.6 AR runs per answer, mean 10.2 AR tokens per run.
  Paired McNemar (HumanEval): ar8 vs pure diffusion 7 vs 5, p=0.77; vs ar32 4 vs 8, p=0.39; vs ar1 7 vs 7, p=1.0; vs AR 12 vs 23, p=0.09. GSM8K: all p >= 0.22.
- **Findings:** shorter AR segments monotonically shift the router toward diffusion (87.5% -> 96.0% -> 99.4% share) and toward pure-diffusion accuracy on HumanEval (76.2% -> 73.8% -> 73.8%, vs 72.6% pure diffusion). The extra AR tokens in the 32-token setting come from the router being *forced* to keep decoding AR, not from it choosing AR. GSM8K is unaffected.
- **Conclusion:** with boundary next-token entropy as the signal, the AR segment length mostly sets how much AR is forced, i.e. a crude knob on the diffusion share; it does not make the router smarter. Confirms E8: the useful decisions must be made inside diffusion blocks.
- **Artifacts:** `runs/router_bench/` (`setting: "entropy:0.5@ar8"`).

### E10. Learned-router infrastructure: SGLang RoutedDecoding verification (2026-10-03)

- **What was built:** `src/router_policy.py` (RouterPolicy MLP on [hidden 3072, entropy, top1, last_dlm] -> p(diffusion), zero-init = p 0.5; same file copied into the fork); HF `LearnedRouter` + eval setting `learned:<ckpt>`; SGLang algorithm `RoutedDecoding` (`patches/sglang-routed.patch`): per-request routing at segment boundaries, AR = 1 token per call (8 per decision), diffusion = seeded 32-token block (threshold 0.9), one causal commit pass per call, router as model submodule `router.*`, per-request JSONL traces (features, action, logp, per-call tokens/NFE); `launch_sglang.sh routed` (POLICY_MODE, TRACE_DIR, ROUTER_CKPT; eager, page size 1).
- **Bug found and fixed:** the fork forces KV page size = block size (32) for every DLLM algorithm except LinearSpec; partial-block commits then corrupt the page table, which made the commit pass non-causal from the second block on (pos-0 logits changed by up to 13 when later positions changed). Added RoutedDecoding to the page-size-1 override; causal check then exact (diff 0.0000) on every call.
- **Parity vs HF (GSM8K 5 + HumanEval 5 prompts, 256 tokens, greedy):** `fixed_ar` vs `ar_generate` identical 7/10 (others share 45-123 tokens first); `fixed_dlm` vs `generate(32, 0.9)` identical 4/10 (others share 46-196 tokens). Divergences after long common prefixes = bf16 kernel near-ties (cf. E3: SGLang LinearSpec 19/30, FastDiffuser 8/30). Note the stock SGLang FastDiffuser does not seed blocks from the causal pass the way HF does; RoutedDecoding does.
- **Router features:** same prompts on both sides, 24 matched decision points: hidden cosine 0.9993-1.0000, scalar features within 0.05; logp recomputed from traced fp16 features equals the server's logp exactly.
- **Router-only weight sync:** `update_weights_from_tensor` with only the 4 `router.*` tensors (CPU tensors, `torch.multiprocessing` sharing strategy `file_system`; the default fd strategy fails with an auth error and crashes the scheduler). Verified: bias -8 pushed -> all decisions AR with logp -0.00034 = log(1 - sigmoid(-8)). Router checkpoints are router-only (~790k params, ~3 MB).
- **Open:** accuracy-level parity on full benchmarks (with the final eval); one client stalled after a weight push (fresh process fine), to watch in the trainer.

### E11. GRPO training of the learned router, run1 (2026-10-03)

- **Question:** can RL on task outcome + compute cost teach the router (frozen backbone) to beat hand-made routers on the accuracy/efficiency trade-off?
- **Setup:** `scripts/train_router_rl.py` (run with `.venv-sglang`), 2 SGLang RoutedDecoding servers (GPUs 2, 3, `policy_mode: sample`). 100 iterations x 32 prompts x 8 rollouts, 512-token budget, AR segment 8, diffusion block 32 / thr 0.9. Data mix gsm8k 0.35 / mbpp 0.15 / kodcode 0.5 (`scripts/rl_data.py`: GSM8K train 7,473; MBPP train+val 464; KodCode-V1 filtered sample 2,395 whose reference solutions pass our sandbox, 80% of 3,000). Reward = correct - 0.1 x forward_passes/tokens. GRPO group advantages, PPO clip 0.2, 2 epochs, entropy 0.01, Adam 3e-4. Router-only checkpoints every 10 iterations (`runs/router_rl/run1/router_iter*.pt`, 3.15 MB).
- **Assumptions:** cost weight 0.1 makes a 1-point accuracy loss outweigh the whole AR-vs-diffusion efficiency gap (AR penalty 0.1 vs diffusion ~0.04); same decoding settings as E5-E9 except per-decision AR segment 8.
- **Sanity checks before the run:** dry run (2 iterations) with 0 failed rollouts and on-policy logp match 6e-8 after a sync; synthetic PPO test (advantage = 1 iff action matches a feature rule) agreement 0.49 -> 0.88 over 600 iterations.
- **Results (100 iterations, 0 failed rollouts, ~45-85 s/iteration):** the router **collapsed to always-diffusion**. Mean p(diffusion) 0.65 (it 0-9) -> 0.95 (10-19) -> 0.998 (30+); policy entropy 0.59 -> 0.006; diffusion token share 0.87 -> 1.00; tokens/forward 1.93 -> ~2.18. Training accuracy shows no trend (GSM8K 0.82-0.93, KodCode 0.45-0.62 per 10-iteration window, noisy across sampled prompts); reward rose only through the cost term.
- **Held-out check (HumanEval, greedy decisions, @ar8):** router_iter0010 and router_iter0100 both = pure diffusion: 72.6% (CI 65.3-78.8), 100% diffusion share, 2.46 tokens/forward, identical to random:1.0; vs AR 7 vs 20, p=0.019.
- **Diagnosis (inferred, not yet measured):** with a greedy backbone the G rollouts of a prompt differ only in routing; most groups are all-correct or all-wrong, so reward differences are just the cost term (~0.01-0.05). GRPO's per-group std normalization inflates these to ~unit advantages, so almost every update favours diffusion and the rare groups where routing changes correctness are outvoted; exploration dies within ~20 iterations.
- **Correction from run2's logging (same data/settings):** groups where routing changes correctness are *not* rare: 22-47% of prompt groups in the first 5 iterations. Revised diagnosis: the remaining majority of groups differ only in cost, and std normalization gave each of them unit-size "more diffusion" advantages, while a correctness flip in a mixed group is credited to all ~15 decisions of the rollout (noisier per-decision signal). A large consistent weak signal outweighed a smaller noisy strong one.

### E12. Router GRPO run2: unnormalized advantages + stronger entropy (2026-10-04)

- **Changes vs E11 (only):** advantage = reward - group mean (`--adv-norm none`); entropy coefficient 0.01 -> 0.05 (equivalent to a pull toward p=0.5). New logs: `mixed_correct_group_frac`, `group_reward_std`.
- **Training (100 iterations, 0 failed rollouts):** no collapse, but no learning either: mean p(diffusion) 0.48-0.52 in every 10-iteration window, policy entropy 0.690-0.693 (max 0.693), diffusion token share ~0.78-0.80, tokens/forward ~1.75-1.80; mixed-correct groups 0.28-0.40 per window (signal present). Final output layer weight norm 0.05 (zero init), bias -0.003: the entropy bonus pinned the policy at p~0.5.
- **Held-out eval (greedy decisions, @ar8):**

  | router | HumanEval acc | diffusion share | tokens/forward | mean length | GSM8K acc (share) |
  |---|---|---|---|---|---|
  | run2 it100 | 76.2% | 79.2% | 2.58 | 208 | 86.7% (90.7%) |
  | run2 it50 | 75.0% | 80.0% | 1.89 | 423 | - |
  | control A: it100 first layer + random-direction output (same norm) | 79.3% | 45.3% | 1.54 | 209 | - |
  | control B: fresh init + random-direction output (same norm) | 75.0% | 70.3% | 2.22 | 208 | - |
  | random:0.8 / 0.9 / 1.0 (E5, E7) | 69.5 / 73.8 / 72.6% | 80 / 91 / 100% | 1.96 / 2.22 / 2.46 | 374 / 395 / 415 | - |
  | AR (E5) | 80.5% | 0% | 1.00 | 210 | 87.3% |

  Paired McNemar (HumanEval): it100 vs random:0.8 20 vs 9, p=0.061; vs random:0.9 p=0.57; vs random:1.0 p=0.38; vs entropy:0.5 p=1.0; vs AR 6 vs 13, p=0.17; vs control A p=0.44; vs control B p=0.84; vs it50 p=0.85. GSM8K vs AR p=0.50.
- **Key finding: the first segment's mode sets the answer's style.** AR first -> the model writes code directly (~200 tokens); diffusion first -> it writes an explanation first (~420 tokens). Holds across routers; for random:0.5 (52% AR-first) AR-first answers are 198 tokens / 79.1% correct vs diffusion-first 448 tokens / 73.1%. HumanEval prompts all end in the same chat-template tokens, so a deterministic router's first decision is effectively a constant: run2 it100 and both controls always start with AR (short answers); run2 it50 and run1 always start with diffusion (long answers).
- **Conclusion:** run2's router did not learn routing (weights barely moved; random-direction controls reproduce its behaviour; it50 vs it100 differ only in the first-decision constant). Its favourable HumanEval numbers come from always choosing AR first. Not significant vs comparable routers. Untested candidate rule suggested by this: AR for the first segment, then diffusion (or entropy routing) afterwards.
- **RL takeaways:** std-normalized GRPO collapses to diffusion (E11); unnormalized advantages with entropy 0.05 does not move the policy at all (E12). Next attempts need a weaker entropy bonus (e.g. 0.01-0.02) or per-decision credit (e.g. reward-to-go / position-aware baselines), and evaluation should report first-decision behaviour separately.

### E13. Router RL with a forced AR start (2026-10-04)

- **Question:** E12 showed the first segment's mode sets answer style and acts as a constant for a deterministic router. If the first 8 tokens are always AR, can RL learn the remaining AR/diffusion switching?
- **Setup:** `runs/router_rl/run3_forcedar8`; as E12 (unnormalized advantages, 100 iterations x 32 prompts x 8 rollouts, 512 tokens, AR segment 8, diffusion block 32 / thr 0.9, cost weight 0.1, lr 3e-4, same data mix) except: `--forced-ar-tokens 8` (RoutedDecoding `forced_ar_tokens`: the router is not consulted and nothing is traced until 8 tokens are generated) and entropy bonus 0.01. Includes the code-review fixes. Eval: `learned:<ckpt>@ar8@first8` (HF `ForcedARStart` wrapper) on GSM8K test + HumanEval.
- **Verified:** server trace for a forced-start request: 8 AR calls of 1 token, no decision record, first decision at token 8.
- **Training (100 iterations, 0 tracebacks):** small movement only: mean p(diffusion) 0.49 -> 0.55 over the run, policy entropy 0.674-0.692 (max 0.693), diffusion token share 0.76 -> 0.80, tokens/forward 1.75 -> 1.79.
- **Eval (greedy decisions, `@ar8@first8`):**

  | benchmark | router | accuracy | 95% CI | diffusion share | tokens/forward | length |
  |---|---|---|---|---|---|---|
  | HumanEval | E13 router | 73.2% | 65.9-79.4 | 86.5% | **3.11** | 209 |
  | GSM8K | E13 router | 87.3% | 85.4-89.0 | 92.9% | 2.43 | 175 |

  Paired McNemar: HumanEval vs AR 8 vs 20, **p=0.036** (worse); vs pure diffusion (block 32) 12 vs 11, p=1.0; vs random:0.8 15 vs 9, p=0.31. GSM8K vs AR 56 vs 57, p=1.0; vs pure diffusion p=1.0.
- **Conclusion:** with the forced AR start, answers stay AR-length (209 tokens) and the router sends ~87% of tokens to diffusion, giving pure-diffusion accuracy on HumanEval at the highest efficiency seen (3.11 tokens/forward vs 2.46) and AR accuracy on GSM8K. Decisions after the start are state-dependent (second segment AR in 64% of HumanEval answers), but learning was small and no random-direction control was run, so how much is learned vs the forced start is open.

### E14. Entropy-bonus grid for router RL (2026-10-04, queued after E13)

- **Question:** E11 (0.01 + std-normalized advantages) collapsed to diffusion; E12 (0.05, unnormalized) never moved. Where between does unnormalized-advantage RL learn?
- **Setup:** three runs `runs/router_rl/grid_ent{0.005,0.01,0.02}`, identical to E12 (no forced AR start, unnormalized advantages) except the entropy bonus. Eval: `learned:<ckpt>@ar8`. `grid_ent0.01` differs from E13 only by the forced AR start.
- **Results:**

  | benchmark | entropy bonus | accuracy | 95% CI | diffusion share | tokens/forward | length |
  |---|---|---|---|---|---|---|
  | HumanEval | 0.005 | 72.0% | 64.6-78.3 | 88.5% | 2.09 | 415 |
  | HumanEval | 0.01 | 71.3% | 64.0-77.7 | 93.2% | 2.31 | 410 |
  | HumanEval | 0.02 | 71.3% | 64.0-77.7 | 83.2% | 2.84 | 208 |
  | GSM8K | 0.005 | 87.3% | 85.4-89.0 | 95.4% | 2.51 | 172 |
  | GSM8K | 0.01 | 87.9% | 86.1-89.6 | 95.3% | 2.54 | 173 |
  | GSM8K | 0.02 | 87.1% | 85.2-88.8 | 91.5% | 2.40 | 174 |

  Training p(diffusion) at the last iteration: 0.53 (0.005), 0.58 (0.01), 0.52 (0.02). The 0.005 and 0.01 routers take diffusion as the first decision on every answer (410-token explanation-first HumanEval answers); the 0.02 router takes AR first on every answer (208-token code-first answers, like E13's forced start). The first decision is a near-constant of the barely-trained policy, so which way it falls is essentially chance. Paired McNemar: HumanEval 0.005 vs AR p=0.009; 0.01 vs AR 25 vs 10 **p=0.017** (worse), vs pure diffusion (block 32) 8 vs 6 p=0.79; GSM8K 0.01 vs AR p=0.55, vs pure diffusion p=0.26. 0.02: HumanEval vs AR 20 vs 5 **p=0.004** (worse), vs pure diffusion 12 vs 14 p=0.85; GSM8K vs AR p=0.85, vs pure diffusion p=0.93. (The 0.02 eval was generated by the queue but never scored; scored by hand 2026-10-05.)
- **Conclusion:** no entropy bonus in 0.005-0.02 yields a router better than pure diffusion; all three are at pure-diffusion accuracy on HumanEval (71-72%, significantly below AR) and at AR accuracy on GSM8K. Lower bonuses let the policy drift toward "more diffusion" (the cost term), not toward placing AR where it helps. The 0.02 run reaches E13-like efficiency (2.84 tokens/forward, short answers) only because its first decision happens to be AR. Sequence-level reward credited to all decisions does not teach where AR helps.

### E15. Fine-grained routing: diffusion block 4, decision after every AR token (2026-10-04)

- **Question:** E8/E9 found the costly mistakes happen inside long (32-token) diffusion blocks. With 4-token diffusion blocks and a router decision after every AR token, does a learned router get the control it needs?
- **Setup:** (1) no-RL baselines at this granularity on GSM8K test + HumanEval: `random:1.0@ar1@blk4` (pure diffusion, block 4), `random:0.5@ar1@blk4`, `entropy:0.5@ar1@blk4` (pure AR is block-independent: reuse E5); (2) RL `runs/router_rl/run4_blk4_ar1`: as E12/E13 (unnormalized advantages, 100 x 32 x 8, 512 tokens, cost weight 0.1, lr 3e-4, same data) with `--block-size 4 --ar-chunk 1`, entropy bonus 0.01, no forced AR start; (3) eval `learned:<ckpt>@ar1@blk4`. Eval records in `runs/router_bench_e15/` (separate dir: the E13/E14 queue writes `runs/router_bench/` concurrently).
- **New plumbing:** `@blk<n>` setting suffix (per-setting diffusion block length), `BLOCK_SIZE` for routed SGLang servers, trainer `--block-size`.
- **SGLang parity at block 4 (GSM8K 5 + HumanEval 5, 256 tokens):** `fixed_dlm` vs HF `generate(block 4, thr 0.9)` identical 5/10 (others share 19-146 tokens); `fixed_ar` vs `ar_generate` identical 8/10.
- **Baselines (no RL; `runs/router_bench_e15/`):**

  | benchmark | router | accuracy | 95% CI | diffusion share | tokens/forward | length |
  |---|---|---|---|---|---|---|
  | HumanEval | pure diffusion, block 4 | 78.0% | 71.1-83.7 | 100% | 1.74 | 211 |
  | HumanEval | random 0.5, block 4 | 76.8% | 69.8-82.6 | 80% | 1.52 | 210 |
  | GSM8K | pure diffusion, block 4 | 88.2% | 86.3-89.8 | 100% | 1.51 | 174 |
  | GSM8K | random 0.5, block 4 | 87.8% | 85.9-89.5 | 80% | 1.37 | 177 |

  Reference: AR HumanEval 80.5% / GSM8K 87.3% (1.00 tokens/forward); pure diffusion at block 32 HumanEval 72.6% (2.46 tokens/forward, 415-token answers). **Block size alone is a strong accuracy/efficiency knob:** block 4 diffusion is within 2.5 points of AR on HumanEval with AR-length answers, at 1.74 tokens/forward. (entropy:0.5 block-4 numbers: see final table.)
- **RL attempt 1 crashed** at iteration 7 (`runs/router_rl/run4_blk4_ar1`): a sandbox container exited with status 1 (runner crash, not OOM), most likely a model output with a lone surrogate character that could not be written as UTF-8; one bad program killed the scoring batch and the trainer. Fixed in `scripts/rl_data.py`: files written with replacement characters, every per-program failure counts as a failed test, and a failing container is retried by bisection so only the offending program is marked wrong (tested: correct / surrogate / fork bomb / wrong -> True / True / False / False). The hand-written queue script also misreported the crash as "exit 0" (`$?` reset by `$(date)` in the echo); the config runner checks exit codes properly.
- **RL relaunched** with the config runner (`configs/experiments/e15_blk4_ar1.yaml`, outputs `runs/experiments/e15_blk4_ar1/`; baseline records copied in so they are not recomputed).
- **Training (100 iterations, ~45 s each):** the router did not move: mean p(diffusion) 0.49-0.53 per 20-iteration window, entropy 0.685-0.692; diffusion share ~0.80, tokens/forward ~1.33.
- **Final table (all block 4, decision after every AR token):**

  | benchmark | router | accuracy | 95% CI | diffusion share | tokens/forward | length |
  |---|---|---|---|---|---|---|
  | HumanEval | learned (E15) | 78.7% | 71.8-84.2 | 81.1% | 1.52 | 213 |
  | HumanEval | pure diffusion | 78.0% | 71.1-83.7 | 100% | 1.74 | 211 |
  | HumanEval | random 0.5 | 76.8% | 69.8-82.6 | 80.0% | 1.52 | 210 |
  | HumanEval | entropy 0.5 | 73.8% | 66.6-79.9 | 98.7% | 1.74 | 216 |
  | GSM8K | learned (E15) | 88.0% | 86.2-89.7 | 80.9% | 1.39 | 177 |
  | GSM8K | pure diffusion | 88.2% | 86.3-89.8 | 100% | 1.51 | 174 |
  | GSM8K | random 0.5 | 87.8% | 85.9-89.5 | 79.9% | 1.37 | 177 |
  | GSM8K | entropy 0.5 | 88.8% | 87.0-90.4 | 95.7% | 1.50 | 173 |

  Paired McNemar (learned vs): HumanEval AR 6 vs 9 p=0.61, pure diffusion 8 vs 7 p=1.0, random 0.5 8 vs 5 p=0.58, entropy 0.5 10 vs 2 **p=0.039**; GSM8K all p >= 0.34.
- **Conclusion:** the learned router stayed at ~50/50 and behaves like random 0.5 routing. At block 4, pure diffusion is already within noise of AR on both benchmarks and is the most efficient option (1.74 / 1.51 tokens/forward), so routing adds nothing here; the gain comes from the smaller block itself. The entropy router is worse on HumanEval at this granularity (73.8%).

### E16. Forced AR start + learned router with block-8 diffusion (2026-10-04, GPUs 0+1)

- **Question:** combine the two things that worked: E13's forced AR start (short, code-first answers, highest efficiency) and smaller diffusion blocks (E15: better diffusion accuracy). Does a learned router on top beat pure diffusion / random routing under the same start and block size?
- **Setup:** first runner-native experiment, `configs/experiments/e16_forced_ar8_blk8.yaml` -> `runs/experiments/e16_forced_ar8_blk8/`. As E13 (entropy bonus 0.01, unnormalized advantages, 100 x 32 x 8, cost weight 0.1) except diffusion block 8 (was 32); AR segment 8, forced AR tokens 8. Eval (greedy, `@ar8@blk8@first8`): learned, random:1.0 (pure diffusion), random:0.5. With equal 8-token AR segments and diffusion blocks, the diffusion token share ~ p(diffusion).
- **Training (100 iterations, 0 failures):** p(diffusion) 0.53 -> 0.57, policy entropy ~0.675 (max 0.693); rollout diffusion share 0.51-0.55, tokens/forward 1.31-1.35.
- **Results (all with forced 8-token AR start, AR segment 8, diffusion block 8):**

  | benchmark | router | accuracy | 95% CI | diffusion share | tokens/forward | length |
  |---|---|---|---|---|---|---|
  | HumanEval | learned (E16) | 75.0% | 67.9-81.0 | 71.1% | 1.85 | 210 |
  | HumanEval | pure diffusion | 75.6% | 68.5-81.5 | 96.1% | 2.58 | 207 |
  | HumanEval | random 0.5 | 78.7% | 71.8-84.2 | 47.5% | 1.43 | 209 |
  | GSM8K | learned (E16) | 88.7% | 86.9-90.3 | 83.6% | 1.78 | 174 |
  | GSM8K | pure diffusion | 88.4% | 86.6-90.0 | 95.5% | 1.97 | 176 |
  | GSM8K | random 0.5 | 88.5% | 86.6-90.1 | 47.5% | 1.32 | 177 |

  Paired McNemar, HumanEval: learned vs AR 12 vs 3 **p=0.035** (worse); vs pure diffusion (block 8) 5 vs 6 p=1.0; vs random 0.5 4 vs 10 p=0.18; vs E13 learned (block 32) p=0.68. Pure diffusion block 8 vs AR p=0.096; random 0.5 vs AR 7 vs 4 p=0.55. GSM8K: learned vs AR 35 vs 53 p=0.069 (learned better, not significant); all other pairs p >= 0.08.
- **Decision pattern (learned):** the router is task-dependent for the first time: the first routed segment (after the forced 8 AR tokens) is AR in 54% of HumanEval answers vs 12% of GSM8K answers, and it uses more AR on code (diffusion share 71% vs 84%), switching ~9 times per HumanEval answer.
- **Conclusion:** the forced start keeps answers AR-length (~210 tokens) as in E13. The learned router again lands on the pure-diffusion accuracy/efficiency line: same accuracy as pure diffusion at block 8 but less efficient (1.85 vs 2.58 tokens/forward on HumanEval). Its extra AR on code is not placed where it helps; random 0.5 (more AR, uniformly placed) scores higher on HumanEval (n.s.). Block 8 pure diffusion with the forced start (75.6%, 2.58 tokens/forward) sits between block 32 (72.6%, 2.46, but long answers) and block 4 (78.0%, 1.74).

### E17. Cold-start span SFT: the model emits <diff> spans itself (2026-10-05)

- **Motivation:** E11-E16: a separate router trained by sequence-level RL never learned *where* AR helps; diffusion block size and the answer's opening mattered more than routing. New direction (user): make switching part of the model's output. Structure data like function calling: thinking / answers are AR ("slow"), tool calls are actions decoded by diffusion ("fast"), marked by `<diff>` ... `</diff>`. Cold-start SFT first, RL (correctness - compute) afterwards on the backbone.
- **Data:** `nvidia/Nemotron-Post-Training-Dataset-v1`, `tool_calling` split, shard 0 (23,851 reasoning-on multi-turn conversations: <think>, tool calls, tool results, final answer). Kept 14,531 with >= 1 tool call and <= 4,096 tokens; 5% held out by uuid hash (13,801 train / 730 held-out); 8,000 train conversations used. Median conversation 2,598 tokens, 1,416 assistant tokens of which ~72 are span tokens (median span 38 tokens): spans are ~5% of assistant tokens, so overall speedup is bounded; E17 tests the mechanism. (`scripts/sft_data.py`, cache `runs/sft_cache/`.)
- **Untrained special tokens (found in the first smoke test):** the special ids `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` (12-17) and the reserved `<SPECIAL_n>` ids have embedding / LM-head rows at initialization (identical norms, head 0.368 for all); the base model writes these markers as plain text pieces (`<`, `function`, `=`; it gives the special `<think>` id probability 0.000). Rendering chat-template text with them as special ids fed the model untrained embeddings at every turn start and tool call; an 8-step smoke adapter then generated garbage (`</think></think>...`, `\x1b\x1b...`). Fix: a copy of the tokenizer without those six added tokens spells them as text (base assistant NLL 1.07 vs 1.78 with special ids); only `<|im_start|>`, `<|im_end|>` and the span markers are special. The span markers' rows start as the mean of the rows of the text pieces of `<diff>` / `</diff>` (instead of the untrained `<tool_call>` rows) and train at lr 1e-3. Segments (assistant turns, tool calls) are tokenized separately so span boundaries fall on token boundaries. The eval parser accepts calls without a `<tool_call>` wrapper (the base model omits it).
- **Format:** chat template renders calls as `<tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>...</tool_call>`; each is wrapped as `<diff><tool_call>...</tool_call></diff>` using reserved single-id tokens `<SPECIAL_18>`/`<SPECIAL_19>` (no vocabulary resize); `<tool_call>` etc. are text. Loss only on assistant turns.
- **Loss (`src/spans.py`, `src/span_model.py`):** layout [clean sequence ; noisy span blocks] with a flex-attention mask: clean tokens causal; noisy tokens see their own 8-token block bidirectionally plus the clean prefix before the block (what the decoder's KV cache holds). Blocks start at the span start (as at inference); the last block is padded with `</diff>` targets. AR loss (token mean) on assistant tokens outside spans incl. predicting `<diff>`, and on span tokens too (`ar_in_spans: true`: the decoder seeds every block with the AR argmax and commits blocks causally; also keeps an AR fallback for RL). Diffusion loss: masked tokens, CE / t (t ~ U(0.001, 1) per block), normalized by noisy positions. Total = AR + diffusion.
- **Parameters:** LoRA r=32 / alpha 64 on q,k,v,o,gate,up,down + trainable embedding and LM-head rows of the two span tokens (see above), 49.4M trainable; checkpoints store only these (`adapter_*.pt`). AdamW 1e-4, warmup 20, cosine to 0.1x, batch 16 conversations, 1 epoch (500 steps), grad clip 1.0, bf16 backbone, gradient checkpointing, 1 GPU per run.
- **Decoding (`src/span_decode.py`):** AR until `<diff>`; then 8-token diffusion blocks (threshold 0.9, first token seeded by AR) against the causal cache until `</diff>` is unmasked with nothing masked before it; truncate after `</diff>`, commit causally, resume AR. `mode=ar` decodes everything AR with the same weights.
- **Mask verification (`scripts/check_span_layout.py`, base model, one held-out conversation):** clean-sequence logits under the span mask are bit-identical to a pure-causal flex forward (noisy tokens never leak into the AR side); flex vs SDPA differs by kernel noise only (argmax agreement 98.8%, same as eager vs SDPA 98.9%). Every noisy block's logits match the decoder's computation (causal prefill + bidirectional block pass): argmax agreement 100% in all 9 blocks.
- **Evaluation (`scripts/eval_toolcall.py`):** one assistant turn per held-out conversation (uuid hash), true history as prompt; generated turn vs reference: exact call match (names + parameter values), name match, call-vs-answer decision, parse rate, diffusion share, tokens/forward. Settings: span SFT decoded with spans / all-AR; AR-only SFT control (same data and budget, no spans); base model AR.
- **Configs:** `configs/sft/e17_span_sft.yaml`, `e17_ar_sft_control.yaml`, `e17_base.yaml` (runner `scripts/run_sft.py`, outputs `runs/sft/<run>/`).
- **Implementation notes:** gradient checkpointing recomputes layers during backward, so flex mode must stay on through backward (first smoke run failed with a recompute mismatch). torch 2.6 + transformers compile flex attention with max-autotune and static shapes, so lengths are padded to multiples of 1,024 (<= 5 shapes; dynamo's recompile limit is 8).
- **Training (500 steps, ~14 s each, GPUs 0 / 1):** span SFT AR loss 0.84 -> 0.69 (mean of first / last 50 steps), diffusion CE on masked span tokens 0.29 -> 0.14; AR-SFT control AR loss 0.80 -> 0.67. Span-run gradient norms start at 15-28 (1/t-weighted diffusion loss) and fall quickly; LoRA and span rows are clipped separately (first launch clipped jointly; restarted after 2 steps).
- **Results (730 held-out turns, 384 of which call tools; greedy, max 2,048 tokens):**

  | model | decoding | exact | 95% CI | exact on call turns | call/answer decision | diffusion share | tokens/forward | length |
  |---|---|---|---|---|---|---|---|---|
  | base | AR | 48.1% | 44.5-51.7 | 38.5% | 73.0% | 0% | 1.00 | 162 |
  | AR-SFT control | AR | 75.8% | 72.5-78.7 | 60.9% | 87.8% | 0% | 1.00 | 482 |
  | span SFT | **spans (diffusion in <diff>)** | 74.9% | 71.7-77.9 | 59.4% | 87.4% | 5.3% | 1.04 | 504 |
  | span SFT | AR (same weights) | 74.5% | 71.2-77.5 | 58.6% | 87.4% | 0% | 1.00 | 504 |

  Paired McNemar: span SFT (spans) vs base 26 vs 222, p < 1e-39 (call turns 22 vs 102); vs AR-SFT control 41 vs 35, p=0.57 (call turns 30 vs 24, p=0.50); vs the same weights decoded AR 3 vs 6, p=0.51 (656/730 outputs identical, 685/730 identical parsed calls).
- **Switching works:** every one of the 416 tool calls the span model generated was inside a `<diff>` span (416 opened, 416 closed; no call written outside a span), and spans appear only around calls. The model learned when to switch from SFT alone.
- **Speedup:** inside spans diffusion produces 3.02 tokens per forward pass (19,634 span tokens in 6,495 passes, incl. one causal commit pass per block), i.e. ~3x fewer passes for the action tokens at no accuracy cost. Per turn the gain is small because spans are 5.3% of generated tokens (turns are ~380 tokens of thinking + a ~47-token call): 1.04 tokens/forward overall, 1.14 on turns that call a tool; HF wall clock on call turns 6,775 s vs 7,062 s AR (-4%; 6 workers per GPU, eager, batch 1, so wall clock is indicative only).
- **Bug fixes during the run:** (1) the plain-marker tokenizer was rebuilt from a 131k-vocab JSON on every call (rendering 730 conversations took ~20 min; now cached per tokenizer, 5 s); (2) one control-eval worker ran out of memory computing full-prompt logits at prefill (6 workers per GPU); the decoder now runs the LM head on the last position only, and the eval resumed (654/730 turns were done).
- **Conclusion:** cold-start SFT teaches the backbone to mark its own actions: with LoRA + 2 token rows, 8k conversations and one epoch, the model emits `<diff>` exactly around tool calls, decodes them by block diffusion at ~3 tokens/pass with the same accuracy as AR decoding and as an AR-only SFT of the same data (+27 points over the base model from the SFT itself). The end-to-end speedup is bounded by the action fraction (5% here). Next: RL on the backbone (reward correct - compute) to make spans longer / thinking shorter, larger diffusion blocks inside spans (block 8 was chosen conservatively), and evaluation with real tool execution (multi-turn).
- **Closed-loop multi-turn rollouts (2026-10-05):** the single-turn eval never shows interleaving (in this data a turn is always `<think>` then calls; thinking and acting alternate across turns). `scripts/rollout_toolcall.py` rolls out the 389 held-out conversations with >= 2 tool-calling turns on the model's own history: after each turn, the dataset's (synthetic) tool results are appended; a rollout continues while the model calls the same functions as the reference (argument differences recorded) and stops at the first different function or decision, or at the end.
  - **Results (SGLang, all 389):** 201 (51.7%) reach the final answer; 3.87 turns per rollout (reference 5.12); on call turns the same functions 82.0%, exact calls 58.0% (779 turns). Stops: answered instead of calling 109, called instead of answering 48, different function 31. Every one of 773 generated calls is inside a `<diff>` span (773 opened / 773 closed). Inside spans 3.06 tokens per forward pass (35,570 tokens / 11,608 passes); whole rollouts 1.04 tokens/pass, diffusion share 6.2%.
  - **HF vs SGLang:** on the 114 conversations the HF reference finished before it was stopped: reached the final answer 50.9% (HF) vs 55.3% (SGLang), same functions 82.8% vs 85.2%, exact 64.4% vs 64.6%. Single-turn parity (24 span turns): 17/24 token-identical outputs, 23/24 identical calls (bf16 numerics, cf. E3).
  - **SGLang serving of span SFT:** new fork algorithm `SpanDecoding` (`python/sglang/srt/dllm/algorithm/span.py`, in `patches/sglang-routed.patch`): AR until the model commits `<diff>`, then 8-token blocks (AR-seeded, threshold 0.9) until a `</diff>` with no mask before it, truncate and commit (page size 1), back to AR; per-call costs traced. Served from a merged export (`scripts/export_merged.py`: LoRA merged, span rows written into embedding / LM head; 7.7 GB, derived and deletable); `MODEL=<export> scripts/launch_sglang.sh span`. With 128 conversations in flight and the radix cache reusing earlier turns, the 389 rollouts (~1,500 turns) took ~13 min on GPU 3, vs an estimated ~2.5 h for the HF decoder (one conversation per worker, 6 workers, full re-prefill each turn). Context is capped per turn at 8,192 tokens (model turns run longer than reference turns).
- **Viewer:** `scripts/build_trace_viewer.py` builds a local page (`runs/sft/e17_span_sft/trace_viewer.html`) with single turns and full rollouts; AR text plain, diffusion blocks highlighted.
- **Artifacts:** `runs/sft/e17_span_sft/` (adapter `train/adapter_final.pt`: LoRA + span rows only, 189 MB per checkpoint; `eval/generations.jsonl`, `eval/summary.json`), `runs/sft/e17_ar_sft_control/`, `runs/sft/e17_base/`.

### E18a. Where could diffusion take over? Diffusability pilot (2026-10-05)

- **Question:** E17 spans come from the data format (tool calls, ~4-6% of assistant tokens). Which other parts of a turn can the model's own block diffusion reproduce cheaply, and do entropy signals predict them?
- **Setup:** 25 held-out turns (8,717 generated tokens), the E17 span-SFT model's own AR outputs (setting `ar`) as target text. For every token position s, simulate the span decoder: KV cache = true prefix, block position 0 seeded with the AR token, 7 masked positions filled by threshold-0.9 unmasking (`src/diffusability.py`; all candidate blocks of a turn denoised together with the span-SFT layout, one forward per iteration). Checked against the real decoder on 30 random blocks: 29/30 identical tokens, 27/30 identical pass counts. A block is "good" if it reproduces the AR tokens exactly within <= 2-3 denoising passes; good blocks are tiled greedily into spans (>= 1 or >= 2 blocks; every span is >= 8 tokens). Predicted cost: 1 AR pass for `<diff>`, denoising passes + 1 commit pass per block, 1 pass per AR token. `scripts/diffusability_pilot.py`, `scripts/diffusability_report.py`; outputs `runs/sft/e17_span_sft/diffusability/` (`pilot.json`, `annotated.md`).
- **Results:** 56.6% of the 8,542 candidate blocks are reproduced exactly, but only 21.3% within <= 3 passes (exact blocks by passes: 1: 883, 2: 451, 3: 487, ..., 7: 1,114; slow exact blocks give no speedup).

  | rule | spans | thinking covered | answer covered | calls covered | predicted tokens/pass |
  |---|---|---|---|---|---|
  | <= 2 passes, >= 1 block | 148 | 13% | 25% | 80% | 1.16 |
  | <= 2 passes, >= 2 blocks | 50 | 4% | 19% | 74% | 1.11 |
  | <= 3 passes, >= 1 block | 184 | 20% | 34% | 88% | 1.19 |
  | <= 3 passes, >= 2 blocks | 72 | 11% | 25% | 86% | 1.15 |

  By structure (share of tokens; good blocks <= 3 passes; exact in 1 pass): prose in thinking 62% / 11% / 3%; prose in final answers 16% / 12% / 4%; markdown list items 12% / 44% / 28%; tool calls 10% / 78% / 52%; JSON 0.1% / 50% / 17%. No code blocks or tables in these turns.
- **What the spans are:** copied facts (coordinates, prices, names), function names and arguments, formulaic phrases (" see. The user is looking for a", "**Subject:**"), list scaffolding; reasoning steps almost never. The labeler independently marks 80-88% of tool-call tokens, a sanity check.
- **Near misses** (6-7 of 8 tokens right) are mostly paraphrases ("I should call" vs "I'll call", "more information" vs "more details"), sometimes errors ("inspire mente"): exact match is too strict for free text.
- **Entropy:** separating good from bad blocks (AUC, lower = good): first-pass masked entropy 0.959, AR entropy summed over the block 0.973 (the latter needs the block's tokens, so it is not available before decoding; masked entropy is, after the first draft pass).
- **Conclusion / next (E18b):** diffusion is reliable on systematic text (tool calls, list items, JSON; code blocks and tables expected, not present here) and rarely on prose. Label spans by structure (always diffusion: tool calls, code blocks, JSON, tables, list items; ~22% of tokens here vs ~4-6% in E17) plus measured prose blocks, ideally with an "AR-plausible" criterion (diffusion block's AR log-likelihood close to AR's own) so paraphrases count; spans >= 1 block. The predicted tokens/pass here (1.15-1.19) is for the untrained model; span SFT on these labels should cut passes.

### Fixes after code review (2026-10-04)

Applied before any further router training (E13 onward); E11/E12 ran without them.
1. **Per-request router state no longer held on the GPU.** RoutedDecoding kept each request's boundary hidden state and full-vocabulary logits (~0.5 MB) on the GPU and only freed them on EOS; requests ending at the length limit or aborted leaked until a 20k-entry LRU cap. State now holds only the router features (hidden on CPU in float32, entropy, top-1; ~12 KB). No GPU memory growth was seen in E12 (66.7/67.1 GB at iteration 8, about the start level), so E11/E12 were not affected in practice.
2. **Cost denominator uses the tokens actually returned.** The trace counts whole committed diffusion blocks, but the scheduler truncates the last block at `max_new_tokens`; the trainer now uses the server's `completion_tokens` (forward passes still counted as spent). E11/E12 slightly underestimated diffusion's cost on truncated rollouts (a small extra pull toward diffusion).
3. **Resume safety.** Request ids carry a per-process run tag and stale trace files are deleted before sending, so a resumed run never mixes in traces from a crashed attempt.
4. **Server shutdown.** Servers start in their own process group; `stop()` sends SIGTERM to the group, escalates to SIGKILL after 60 s and never raises.
5. **Full-precision traces.** Decision features are now traced as float32 (was fp16): the trainer recomputes exactly what the server acted on (mismatch 6e-8 = float32 CPU-vs-GPU kernel difference).
   E11/E12 do **not** need retraining for this: their logged trainer-vs-server logp mismatch was max 3.0e-5 (run1, mean 9.3e-6) and max 6.0e-7 (run2), i.e. PPO ratio errors of ~0.003% against a 0.2 clip range. Their failures come from the objective (std normalization; entropy bonus), not precision.

Verified with a 2-iteration run + resume to iteration 3 at a 64-token limit: 48 rollouts, 0 failed, resume OK, 0 leftover server processes.

## 6. Reference numbers and related work

- **NVIDIA 3B table (paper, not reproduced by us except GSM8K):** AR avg 55.50 @ TPF 1.00; Diff. 52.90 @ 1.91; Linear SS 55.00 @ 4.36; Quad SS 55.80 @ 5.42. HumanEval AR 76.22 / Diff. 74.39 / Linear SS 75.00. Quad SS is not in the released code.
- **Training data (tech report):** Ministral3 base -> 1T tokens pure AR + 300B joint AR+diffusion (alpha=0.3) on the Nemotron Nano 2 pretraining set; SFT 45B tokens from the Nemotron 3 Super SFT set (joint loss, alpha=0.3). Possible SWE/agentic contamination unverified.
- **LinearSpec** (in released code): fixed rule, draft block with diffusion, verify causally, accept matching prefix + bonus token. Not a learned router.
- **LearnedSampler** (staged SGLang branch, unmerged, no public checkpoint): small transformer scoring which masked tokens to commit inside diffusion; inputs are top-k probs/entropy/token features, not the backbone hidden state.
- **S2D2** (arXiv 2603.25702): training-free self-speculation for block diffusion; routing policies decide when to verify (min-span, score-threshold, hysteresis, UCB bandit); bandit was not the best policy; entropy-based estimator gave the best accuracy. Tested on SDAR, Fast-dLLM v2, LLaDA2.1, not Nemotron.
- **SDAR:** AR-initialised block-diffusion model; single decoding mode, fixed block per checkpoint; no inference-time switching.
- **Looped Diffusion Language Models / LoopMDM** (arXiv 2605.26106, KAIST/KRAFTON/UC Berkeley, 2026-05): masked-diffusion transformer split into head / looped mid-block / tail; a few early-to-middle layers looped S times (S ~ U{1..S_max} in training). Matches non-looped NLL with up to 3.3x fewer training FLOPs; GSM8K up to +8.5 over same size and above a deeper iso-per-step-FLOP baseline. Loop gains peak at intermediate denoising timesteps and with few denoising steps; mask-to-mask attention grows with loops (masks as a parallel workspace); stopping when the hidden-state change falls below a threshold cuts loops 12 -> ~5. 125-170M models trained from scratch. Relevance: cheaper per-pass refinement inside our spans (we pay a full pass per ~3 span tokens).
- **Recursive Masked Diffusion Models / R-MDM** (arXiv 2606.18022, EPFL/Cambridge, 2026-06): whole denoiser looped L times per denoising step, logits supervised after every loop, loop-index embedding, reverse curriculum on L. Loops substitute for denoising steps (Sudoku: 3 loops x 5 steps = baseline at 40 steps, 2.7x fewer passes) and for parameters on structured tasks (Sudoku, Countdown); worse than the baseline on character-level text (Text8). <= ~50M parameters.
- **Looped Diffusion Transformer** (arXiv 2609.40305, 2026-09): looping for text-to-image diffusion; not language.
- **LaDiR** (arXiv 2510.04573, UCSD/Apple, ICLR 2026): latent diffusion for reasoning. A VAE encodes each CoT sentence into a block of continuous latent tokens (~4 latents per ~22 text tokens); the LLM backbone denoises latent blocks with flow matching (bidirectional within a block, causal across blocks), predicts continue-thinking vs start-answer, then writes the answer AR. Teacher-forced training, then rollout training that backpropagates answer loss into generated latents; diversity guidance at inference. Better pass@1 and much better pass@100 on math/code; ~AR latency at 10 denoising steps. Relevance: same block-causal structure as our spans, but denoises compressed continuous thoughts instead of masked tokens (speed from compression rather than parallel decoding). Follow-ups: LaDi-RL (2602.01705), Uni-LaDiR (2609.19878).
- **None of the looping work retrofits looping onto a pretrained multi-billion-parameter diffusion LM**; doing so for Nemotron 3B would need real training (not LoRA), and R-MDM's text result is negative.

## 7. Open questions and next steps

1. Does entropy predict where diffusion fails? Offline check: at each block start in AR outputs, compare candidate signals (next-token entropy, rolling entropy over recent AR tokens, mean entropy over a diffusion draft) with the draft's match length vs AR. E7: boundary next-token entropy is weak and saturates.
2. Decide inside diffusion blocks (E8): use the commit pass's per-position causal entropy / AR disagreement to cut a block and hand over to AR; also diffusion block 8. (Per-token AR checks alone, E8, just give pure diffusion.)
3. Multiple router seeds for random baselines (A4) so the random curve is smooth enough to compare against; report measured diffusion share instead of p.
4. Oracle headroom vs LinearSpec (4.36 TPF) before investing in RL.
