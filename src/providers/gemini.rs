//! Google Gemini provider — Gemini API (Flash, Pro, Flash Lite).
//!
//! Free tier available via Google AI Studio: https://aistudio.google.com/apikey
//! Free tier limits (as of 2025):
//!   - gemini-2.5-flash: 500 RPD, 10 RPM
//!   - gemini-2.5-pro:   50 RPD, 5 RPM
//!   - gemini-2.0-flash-lite: 1500 RPD, 30 RPM

use crate::config::AiConfig;
use crate::error::AiError;
use crate::models::*;
use crate::util::backoff_duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

// =============================================================================
// Gemini Request / Response Types (internal)
// =============================================================================

#[derive(Debug, Serialize)]
struct GeminiRequest {
    /// The conversation contents (user + model turns).
    contents: Vec<GeminiContent>,

    /// Optional system instruction (replaces the old system role approach).
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,

    /// Generation configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    role: String, // "user" or "model"
    parts: Vec<GeminiPart>,
}

/// A Gemini content part — either text or inline image data.
///
/// Uses `#[serde(untagged)]` so that each variant serializes to its own
/// shape without a wrapping tag:
///   - Text:  `{ "text": "..." }`
///   - Image: `{ "inline_data": { "mime_type": "...", "data": "..." } }`
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    InlineData { inline_data: GeminiInlineData },
}

/// Inline image data for Gemini vision requests.
#[derive(Debug, Serialize)]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiSystemPart>,
}

/// System instruction parts are always text-only.
#[derive(Debug, Serialize)]
struct GeminiSystemPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,

    max_output_tokens: u32,

    /// Instruct the model to return JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
}

// --- Response ---

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
    #[serde(rename = "modelVersion")]
    model_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContentResponse>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiContentResponse {
    parts: Option<Vec<GeminiPartResponse>>,
}

#[derive(Debug, Deserialize)]
struct GeminiPartResponse {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u32>,
}

/// An API error returned by Gemini in the response body.
#[derive(Debug, Deserialize)]
struct GeminiErrorResponse {
    error: Option<GeminiErrorDetail>,
}

#[derive(Debug, Deserialize)]
struct GeminiErrorDetail {
    #[allow(dead_code)]
    code: Option<u16>,
    message: Option<String>,
    #[allow(dead_code)]
    status: Option<String>,
}

// =============================================================================
// Provider Function
// =============================================================================

/// Send a chat request to the Google Gemini API.
///
/// Uses the `generateContent` endpoint.
///
/// Retry logic:
/// - 429 Too Many Requests — respects `retry-after` or `retry-delay` header,
///   falls back to exponential backoff (important for the free tier's low RPM)
/// - 503 Service Unavailable — exponential backoff
/// - Network timeouts — exponential backoff
pub async fn chat_gemini(
    http: &Client,
    config: &AiConfig,
    request: &AiChatRequest,
    variant: GeminiModel,
) -> Result<AiChatResponse, AiError> {
    let api_key = config
        .gemini_api_key
        .as_ref()
        .ok_or_else(|| AiError::ProviderNotConfigured("GEMINI_API_KEY not set".to_string()))?;

    let model_id = variant.model_id();

    // Build the URL: POST .../models/{model}:generateContent?key={api_key}
    let url = format!(
        "{}/{}:generateContent?key={}",
        GEMINI_API_BASE, model_id, api_key
    );

    // Map gateway messages to Gemini contents.
    // Gemini uses "user" / "model" (not "assistant").
    let contents: Vec<GeminiContent> = request
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                AiRole::User => "user".to_string(),
                AiRole::Assistant => "model".to_string(),
            };

            if m.images.is_empty() {
                // Simple text-only message
                GeminiContent {
                    role,
                    parts: vec![GeminiPart::Text {
                        text: m.content.clone(),
                    }],
                }
            } else {
                // Multimodal message: images first, then text
                let mut parts: Vec<GeminiPart> = m
                    .images
                    .iter()
                    .filter_map(|img| match img {
                        AiContentPart::Image { data, media_type } => Some(GeminiPart::InlineData {
                            inline_data: GeminiInlineData {
                                mime_type: media_type.clone(),
                                data: data.clone(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();

                if !m.content.is_empty() {
                    parts.push(GeminiPart::Text {
                        text: m.content.clone(),
                    });
                }

                GeminiContent { role, parts }
            }
        })
        .collect();

    // System instruction
    let system_text = match request.response_format {
        AiResponseFormat::Json => {
            let base = request.system.clone().unwrap_or_default();
            Some(format!(
                "{}\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown fences, no explanation, just pure JSON.",
                base
            ))
        }
        AiResponseFormat::Text => request.system.clone(),
    };

    let system_instruction = system_text.map(|text| GeminiSystemInstruction {
        parts: vec![GeminiSystemPart { text }],
    });

    // Generation config
    let response_mime_type = match request.response_format {
        AiResponseFormat::Json => Some("application/json".to_string()),
        AiResponseFormat::Text => None,
    };

    let generation_config = Some(GeminiGenerationConfig {
        temperature: request.temperature,
        max_output_tokens: request.max_tokens,
        response_mime_type,
    });

    let gemini_request = GeminiRequest {
        contents,
        system_instruction,
        generation_config,
    };

    debug!("Sending chat request to Gemini model '{}'", model_id);

    // -------------------------------------------------------------------------
    // Retry loop
    // -------------------------------------------------------------------------
    let max_attempts = config.max_retries + 1;
    let mut last_err: Option<AiError> = None;

    for attempt in 1..=max_attempts {
        let resp = http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&gemini_request)
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                let err = if e.is_timeout() {
                    AiError::Timeout(format!("Gemini request timed out: {}", e))
                } else {
                    AiError::NetworkError(e.to_string())
                };
                if attempt < max_attempts {
                    let wait = backoff_duration(attempt);
                    warn!(
                        "Gemini network error attempt {}/{}, retrying in {}s: {}",
                        attempt,
                        max_attempts,
                        wait.as_secs(),
                        e
                    );
                    tokio::time::sleep(wait).await;
                }
                last_err = Some(err);
                continue;
            }
        };

        let status = resp.status();

        // 429 — Rate limited (very common on the free tier!)
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            // Gemini may return retry delay in the response body or header
            let retry_after_header = resp
                .headers()
                .get("retry-after")
                .or_else(|| resp.headers().get("x-ratelimit-reset-after"))
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());

            let body = resp.text().await.unwrap_or_default();

            // Try to parse retry delay from Gemini error body
            let retry_after_body = parse_retry_delay_from_body(&body);
            let retry_after = retry_after_header.or(retry_after_body);

            if attempt < max_attempts {
                // Free tier: default to 15s wait if no hint given (respects 10 RPM limit)
                let wait_secs = retry_after.unwrap_or(15).min(90);
                warn!(
                    "Gemini rate limited (429) attempt {}/{}, waiting {}s (free tier limit)",
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

        // 503 — Service unavailable
        if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            let body = resp.text().await.unwrap_or_default();
            if attempt < max_attempts {
                let wait = backoff_duration(attempt);
                warn!(
                    "Gemini service unavailable (503) attempt {}/{}, retrying in {}s",
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

        // Any other non-2xx → hard error
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            // Try to extract a human-readable message from Gemini's error JSON
            let message = extract_gemini_error_message(&body)
                .map(|msg| format!("{} — {}", status, msg))
                .unwrap_or_else(|| body.clone());
            return Err(AiError::ApiError {
                status: status.as_u16(),
                body: message,
            });
        }

        // Success path
        if attempt > 1 {
            info!("Gemini succeeded on attempt {}/{}", attempt, max_attempts);
        }

        let parsed: GeminiResponse = resp
            .json()
            .await
            .map_err(|e| AiError::ParseError(format!("Failed to parse Gemini response: {}", e)))?;

        let content = extract_text_content(&parsed)?;

        // Determine which model actually responded (Gemini returns modelVersion)
        let model_used = parsed.model_version.unwrap_or_else(|| model_id.to_string());

        let usage = parsed.usage_metadata.map(|u| AiUsage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
        });

        return Ok(AiChatResponse {
            content,
            model_used,
            provider: AiProvider::Gemini,
            usage,
        });
    }

    Err(last_err.unwrap_or_else(|| {
        AiError::NetworkError("Gemini request failed after all retries".to_string())
    }))
}

// =============================================================================
// Internal Helpers
// =============================================================================

/// Extract the text content from a Gemini response.
fn extract_text_content(response: &GeminiResponse) -> Result<String, AiError> {
    let candidate = response
        .candidates
        .as_ref()
        .and_then(|c| c.first())
        .ok_or_else(|| AiError::ParseError("Gemini returned no candidates".to_string()))?;

    // Check for safety blocks or other non-STOP finish reasons
    if let Some(ref reason) = candidate.finish_reason {
        match reason.as_str() {
            "SAFETY" => {
                return Err(AiError::ApiError {
                    status: 200,
                    body: "Gemini blocked the response due to safety filters".to_string(),
                });
            }
            "RECITATION" => {
                return Err(AiError::ApiError {
                    status: 200,
                    body: "Gemini blocked the response due to recitation policy".to_string(),
                });
            }
            "MAX_TOKENS" | "STOP" | "FINISH_REASON_UNSPECIFIED" | "" => {
                // Normal finish reasons — continue
            }
            other => {
                warn!("Gemini unexpected finish reason: {}", other);
            }
        }
    }

    let text = candidate
        .content
        .as_ref()
        .and_then(|c| c.parts.as_ref())
        .and_then(|p| p.first())
        .and_then(|p| p.text.clone())
        .unwrap_or_default();

    Ok(text)
}

/// Try to extract a retry delay (in seconds) from Gemini's error body.
///
/// Gemini sometimes embeds retry information in the error details, e.g.:
/// `{"error": {"message": "... retry after 30s ..."}}`
fn parse_retry_delay_from_body(body: &str) -> Option<u64> {
    // Quick heuristic: look for "retry after Xs" in the message
    let lower = body.to_lowercase();
    if let Some(pos) = lower.find("retry after ") {
        let after = &lower[pos + 12..];
        let secs_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(secs) = secs_str.parse::<u64>() {
            return Some(secs);
        }
    }
    None
}

/// Try to extract a human-readable message from a Gemini error JSON body.
fn extract_gemini_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<GeminiErrorResponse>(body)
        .ok()
        .and_then(|e| e.error)
        .and_then(|e| e.message)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_model_ids() {
        assert_eq!(GeminiModel::Flash.model_id(), "gemini-2.5-flash");
        assert_eq!(GeminiModel::Pro.model_id(), "gemini-2.5-pro");
        assert_eq!(GeminiModel::FlashLite.model_id(), "gemini-2.0-flash-lite");
    }

    #[test]
    fn test_parse_retry_delay_from_body_found() {
        let body = r#"{"error": {"message": "Quota exceeded, retry after 30s."}}"#;
        assert_eq!(parse_retry_delay_from_body(body), Some(30));
    }

    #[test]
    fn test_parse_retry_delay_from_body_not_found() {
        let body = r#"{"error": {"message": "Something went wrong"}}"#;
        assert_eq!(parse_retry_delay_from_body(body), None);
    }

    #[test]
    fn test_extract_gemini_error_message() {
        let body = r#"{"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}"#;
        let msg = extract_gemini_error_message(body);
        assert_eq!(msg, Some("Resource exhausted".to_string()));
    }

    #[test]
    fn test_extract_gemini_error_message_invalid_json() {
        let body = "not json at all";
        assert_eq!(extract_gemini_error_message(body), None);
    }

    #[test]
    fn test_extract_text_content_safety_block() {
        let response = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: None,
                finish_reason: Some("SAFETY".to_string()),
            }]),
            usage_metadata: None,
            model_version: None,
        };
        let result = extract_text_content(&response);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("safety"));
    }

    #[test]
    fn test_extract_text_content_success() {
        let response = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContentResponse {
                    parts: Some(vec![GeminiPartResponse {
                        text: Some("Hello from Gemini!".to_string()),
                    }]),
                }),
                finish_reason: Some("STOP".to_string()),
            }]),
            usage_metadata: None,
            model_version: Some("gemini-2.5-flash".to_string()),
        };
        let text = extract_text_content(&response).unwrap();
        assert_eq!(text, "Hello from Gemini!");
    }

    #[test]
    fn test_extract_text_content_no_candidates() {
        let response = GeminiResponse {
            candidates: None,
            usage_metadata: None,
            model_version: None,
        };
        assert!(extract_text_content(&response).is_err());
    }
}
