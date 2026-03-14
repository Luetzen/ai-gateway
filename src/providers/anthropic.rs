//! Anthropic provider — Claude API (Haiku, Sonnet, Opus).

use crate::config::AiConfig;
use crate::error::AiError;
use crate::models::*;
use crate::util::backoff_duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

// =============================================================================
// Anthropic Request/Response Types (internal)
// =============================================================================

#[derive(Debug, Serialize)]
struct AnthropicChatRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicChatResponse {
    content: Vec<AnthropicContent>,
    model: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

// =============================================================================
// Provider Functions
// =============================================================================

/// Send a chat request to the Anthropic API.
///
/// Includes retry logic for transient errors:
/// - 429 Too Many Requests — respects retry-after header
/// - 503 Service Unavailable — exponential backoff
/// - 529 Overloaded — exponential backoff
pub async fn chat_anthropic(
    http: &Client,
    config: &AiConfig,
    request: &AiChatRequest,
    variant: AnthropicModel,
) -> Result<AiChatResponse, AiError> {
    let api_key = config
        .anthropic_api_key
        .as_ref()
        .ok_or_else(|| AiError::ProviderNotConfigured("ANTHROPIC_API_KEY not set".to_string()))?;

    let model_id = variant.model_id();

    let messages: Vec<AnthropicMessage> = request
        .messages
        .iter()
        .map(|m| AnthropicMessage {
            role: m.role.to_string(),
            content: m.content.clone(),
        })
        .collect();

    // If JSON output is requested, amend the system prompt to instruct Claude
    let system = match request.response_format {
        AiResponseFormat::Json => {
            let base = request.system.clone().unwrap_or_default();
            Some(format!(
                "{}\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown fences, no explanation, just pure JSON.",
                base
            ))
        }
        AiResponseFormat::Text => request.system.clone(),
    };

    let anthropic_request = AnthropicChatRequest {
        model: model_id.to_string(),
        max_tokens: request.max_tokens,
        system,
        messages,
        temperature: request.temperature,
    };

    debug!("Sending chat request to Anthropic model '{}'", model_id);

    // Retry loop for Anthropic (429, 529, 503)
    let max_attempts = config.max_retries + 1;
    let mut last_err: Option<AiError> = None;

    for attempt in 1..=max_attempts {
        let resp = http
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&anthropic_request)
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                if e.is_timeout() {
                    last_err = Some(AiError::Timeout(format!(
                        "Anthropic request timed out: {}",
                        e
                    )));
                } else {
                    last_err = Some(AiError::NetworkError(e.to_string()));
                }
                if attempt < max_attempts {
                    let wait = backoff_duration(attempt);
                    warn!(
                        "Anthropic network error attempt {}/{}, retrying in {}s",
                        attempt,
                        max_attempts,
                        wait.as_secs()
                    );
                    tokio::time::sleep(wait).await;
                }
                continue;
            }
        };

        let status = resp.status();

        // 429 — Rate limited
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            let body = resp.text().await.unwrap_or_default();

            if attempt < max_attempts {
                let wait_secs = retry_after.unwrap_or(10).min(90);
                warn!(
                    "Anthropic rate limited (429) attempt {}/{}, retry-after={}s",
                    attempt, max_attempts, wait_secs
                );
                tokio::time::sleep(Duration::from_secs(wait_secs)).await;
                last_err = Some(AiError::RateLimited {
                    retry_after_secs: retry_after,
                    body,
                });
                continue;
            }
            return Err(AiError::RateLimited {
                retry_after_secs: retry_after,
                body,
            });
        }

        // 529/503 — Overloaded
        if status.as_u16() == 529 || status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            let body = resp.text().await.unwrap_or_default();
            if attempt < max_attempts {
                let wait = backoff_duration(attempt);
                warn!(
                    "Anthropic overloaded ({}) attempt {}/{}, retrying in {}s",
                    status,
                    attempt,
                    max_attempts,
                    wait.as_secs()
                );
                tokio::time::sleep(wait).await;
                last_err = Some(AiError::Overloaded(body));
                continue;
            }
            return Err(AiError::Overloaded(body));
        }

        // Non-2xx hard error
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(AiError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        // Success
        if attempt > 1 {
            info!(
                "Anthropic succeeded on attempt {}/{}",
                attempt, max_attempts
            );
        }

        let parsed: AnthropicChatResponse = resp.json().await.map_err(|e| {
            AiError::ParseError(format!("Failed to parse Anthropic response: {}", e))
        })?;

        let content = parsed
            .content
            .first()
            .and_then(|c| c.text.clone())
            .unwrap_or_default();

        return Ok(AiChatResponse {
            content,
            model_used: parsed.model.unwrap_or_else(|| model_id.to_string()),
            provider: AiProvider::Anthropic,
            usage: parsed.usage.map(|u| AiUsage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: match (u.input_tokens, u.output_tokens) {
                    (Some(i), Some(o)) => Some(i + o),
                    _ => None,
                },
            }),
        });
    }

    Err(last_err.unwrap_or_else(|| {
        AiError::NetworkError("Anthropic request failed after all retries".to_string())
    }))
}
