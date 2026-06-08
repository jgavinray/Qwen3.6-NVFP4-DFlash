# OpenClaw Integration

This branch serves Qwen3.6-27B text-only aliases from the RTX PRO 6000 vLLM
backend:

- `qwen36-27b`
- `qwen36-27b-fast`
- `qwen36-27b-deep`

Use `qwen36-27b-fast` for greedy agentic/code traffic and `qwen36-27b-deep` for
sampled creative traffic.

## Provider Shape

Point OpenClaw at the OpenAI-compatible vLLM endpoint:

```json5
{
  models: {
    providers: {
      vllm: {
        baseUrl: "http://<rtx6000-host>:8000/v1",
        apiKey: "not-needed",
        api: "openai-completions",
        models: [
          {
            id: "qwen36-27b-fast",
            name: "Qwen3.6-27B Fast",
            reasoning: true,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 262144,
            contextTokens: 245760,
            maxTokens: 32768,
            params: { temperature: 0 }
          },
          {
            id: "qwen36-27b-deep",
            name: "Qwen3.6-27B Deep",
            reasoning: true,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 262144,
            contextTokens: 245760,
            maxTokens: 32768,
            params: { temperature: 0.7, top_p: 0.95 }
          }
        ]
      }
    }
  }
}
```

Keep request `max_tokens` at least `2048` for thinking-enabled workloads. The
compose default disables thinking for latency-oriented chat.
