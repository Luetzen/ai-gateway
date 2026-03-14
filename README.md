# ai-gateway

> Provider-agnostic AI gateway for Rust — supports LM Studio (local), Anthropic Claude (cloud), and Google Gemini (cloud) with automatic fallback.

## Overview

`ai-gateway` is a Rust crate that provides a unified interface for multiple AI providers. Write your AI logic once, switch providers with a single enum variant. Built-in retry logic, rate-limit handling, and automatic provider fallback included.

## Providers

| Provider | Type | Models |
|----------|------|--------|
| **LM Studio** | Local (OpenAI-compatible) | Any loaded GGUF model |
| **Anthropic** | Cloud | Haiku, Sonnet, Opus |
| **Google Gemini** | Cloud | Flash, Pro, Flash Lite |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ai-gateway = { git = "https://github.com/Luetzen/ai-gateway.git" }
```

Pinned to a specific tag:

```toml
[dependencies]
ai-gateway = { git = "https://github.com/Luetzen/ai-gateway.git", tag = "v0.1.0" }
```

### Usage

```rust
use ai_gateway::{AiClient, AiChatRequest, AiMessage, AiModel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = AiClient::from_env();

    let request = AiChatRequest {
        model: AiModel::Auto,
        system: Some("You are a helpful assistant.".to_string()),
        messages: vec![AiMessage::user("Hello!")],
        max_tokens: 1024,
        temperature: None,
        response_format: Default::default(),
    };

    let response = client.chat(request).await?;
    println!("{}", response.content);

    Ok(())
}
```

### Choosing a Provider

```rust
// Auto mode — tries Local → Anthropic → Gemini
let model = AiModel::Auto;

// Specific local model
let model = AiModel::Local("qwen3-30b-a3b".to_string());

// Anthropic Claude
use ai_gateway::AnthropicModel;
let model = AiModel::Cloud(AnthropicModel::Sonnet);

// Google Gemini
use ai_gateway::GeminiModel;
let model = AiModel::Gemini(GeminiModel::Flash);
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LM_STUDIO_URL` | For local AI | — | LM Studio base URL (e.g. `http://192.168.1.121:1234`) |
| `ANTHROPIC_API_KEY` | For Anthropic | — | Anthropic API key |
| `GEMINI_API_KEY` | For Gemini | — | Google Gemini API key |
| `AI_DEFAULT_LOCAL_MODEL` | No | — | Default local model name |
| `AI_DEFAULT_CLOUD_MODEL` | No | `sonnet` | `haiku`, `sonnet`, or `opus` |
| `AI_DEFAULT_GEMINI_MODEL` | No | `flash` | `flash`, `pro`, or `flash_lite` |
| `AI_AUTO_LOAD_MODELS` | No | `true` | Auto-load models in LM Studio |
| `AI_REQUEST_TIMEOUT_SECS` | No | `120` | Request timeout in seconds |
| `AI_MAX_RETRIES` | No | `3` | Max retries for transient errors |

At least one provider must be configured.

## Auto Mode

When using `AiModel::Auto`, the client cascades through providers in order:

1. **LM Studio** — if `LM_STUDIO_URL` is set and reachable
2. **Anthropic** — if `ANTHROPIC_API_KEY` is set
3. **Gemini** — if `GEMINI_API_KEY` is set

If a provider fails, the next one is tried automatically.

## Features

- **Unified API** — one `chat()` method for all providers
- **Auto fallback** — local AI offline? Cloud takes over seamlessly
- **Retry logic** — exponential backoff for 429, 503, 529 errors
- **Rate-limit handling** — respects `retry-after` headers
- **Auto-load models** — LM Studio models load on demand
- **JSON mode** — request structured JSON output from any provider
- **Utility functions** — `strip_code_fences()` and `parse_ai_json::<T>()` included

## License

MIT — see [LICENSE](LICENSE) for details.