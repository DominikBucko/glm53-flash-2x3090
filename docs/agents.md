# Agents

GLM-5.3-Flash is a reasoning model. Served naively behind a typical agent loop at 3 bits, it tends to stop
reasoning and repeat tool calls. The default profile includes two serving-side fixes and the repository ships
two more as experimental switches. None of them changes the weights.

All results below are single runs of LBX AgentBench v0.2, an internal 15-task suite of synthetic engineering
repositories (debugging, development, retrieval/ML, LLM-serving, incident and rollout tasks; no production data).
Only aggregates are published; fixtures and traces are not. Settings: `reasoning_effort=xhigh`, temperature 0,
32 tool steps, 16,384 output tokens per turn, the runner's own tools (list/read/search/edit/diff/run tests). A
strict pass needs visible and hidden tests, the required discovery, an agent-run test, a diff inspection and an
evidence-based handoff. These runs used the native research build of this runtime on the validation machine, not
the published image; one run is a smoke test, not a conclusion.

## Default profile

**Bounded reasoning.** Every request gets a thinking budget of 4,096 tokens
(`VLLM_DEFAULT_THINKING_TOKEN_BUDGET`). When it runs out, the runtime forces
"I have thought about this enough; I will now act on my best plan." followed by `</think>`, so a turn cannot spend
its whole output budget thinking. `reasoning_effort` accepts OpenAI-style names: `minimal`/`low` map to GLM's
`low`, `medium`/`high` to `high`, `xhigh`/`max` to `max` (the default).

**Preserved reasoning** (`GLM53_PRESERVE_REASONING_TURNS=4`). Agent clients usually send earlier assistant turns
back as content plus tool calls only. GLM's chat template renders those turns as empty `<think></think>`
blocks and the model imitates them: in the first full run it stopped reasoning and looped on 6-token tool calls.
The server now keeps each tool-calling response's reasoning (keyed by tool-call id, in memory, at most 8,192
entries) and puts it back into the last 4 assistant turns of later requests in the same conversation. Clients that
already send `reasoning` are left alone.

| Run (same runner and settings) | Strict passes | Mean score |
|---|---:|---:|
| Reasoning dropped by the client (stock behaviour), first 3 tasks | stopped after looping | — |
| Preserve last 4 turns + prefix caching (default profile) | **9 / 15** | **94.6** |
| Preserve all turns + prefix caching | 3 / 15 | 45.8 |

Keeping every turn was much worse: it also preserves the model's own empty first turns. For reference, the same
runner scored Qwen3.8-Flash-Next (INT4 + MTP3, [qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090))
9 / 15, mean 82.1.

With prefix caching, a tool-only turn took 7.6 s (median) at 72.9% prefix-cache hits; decode ran at 37.7 tok/s
across 257 calls.

## Experimental switches (`configs/agent-experimental.env`)

**Strict tool grammar** (`GLM53_STRICT_TOOL_GRAMMAR=1`). At 3.05 bpw the model occasionally emits a tool call
whose name does not exist (`replace_file_part`, `run_test`, a garbled `replace�_in_file`) or a Python-style
call. The parser leaves it in the content, the client sees no tool call and the task ends, usually with the
workspace half-edited. vLLM can constrain tool calls with xgrammar's `glm_4_7` grammar but only does so for tools
marked `strict`; this switch applies it to every `tool_choice=auto` request. After `</think>` free text stays
allowed, but `<tool_call>` must continue with one of the request's tool names and arguments matching its schema
(any key order). Replays of the failing requests produced valid calls; decode speed and free-text answers were
unchanged. No full benchmark run with it has completed yet.

**Minimum reasoning** (`VLLM_MIN_THINKING_TOKENS=64`). Even with preserved reasoning, about half of all turns had
zero reasoning tokens. This switch bans `</think>`, `<tool_call>` and the end-of-turn tokens for the first 64
reasoning tokens of every turn. It removed zero-reasoning turns completely (0 of 231 turns) but made the run 1.75x
longer and did not improve the score (8 / 15, mean 85.0; two tasks were lost to the malformed tool calls above).
