//! LM Studio provider — OpenAI-compatible API.

use crate::config::AiConfig;
use crate::error::AiError;
use crate::models::*;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

// =============================================================================
// OpenAI-Compatible Request/Response Types (internal)
// =============================================================================

#[derive(Debug, Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAiResponseFormat>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
    model: Option<String>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoiceMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModelsResponse {
    data: Vec<OpenAiModelEntry>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModelEntry {
    id: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct LmStudioLoadRequest {
    model: String,
}

// =============================================================================
// Provider Functions
// =============================================================================

/// Send a chat request to LM Studio (OpenAI-compatible endpoint).
pub async fn chat_local(
    http: &Client,
    config: &AiConfig,
    request: &AiChatRequest,
    model_name: &str,
) -> Result<AiChatResponse, AiError> {
    let base_url = config
        .lm_studio_url
        .as_ref()
        .ok_or_else(|| AiError::ProviderNotConfigured("LM_STUDIO_URL not set".to_string()))?;

    // Auto-load model if enabled
    if config.auto_load_models {
        if let Err(e) = ensure_model_loaded(http, config, model_name).await {
            warn!(
                "Auto-load model '{}' failed: {} — trying request anyway",
                model_name, e
            );
        }
    }

    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    // Build messages: prepend system message if present
    let mut messages = Vec::with_capacity(request.messages.len() + 1);
    if let Some(ref system) = request.system {
        messages.push(OpenAiMessage {
            role: "system".to_string(),
            content: system.clone(),
        });
    }
    for msg in &request.messages {
        messages.push(OpenAiMessage {
            role: msg.role.to_string(),
            content: msg.content.clone(),
        });
    }

    let response_format = match request.response_format {
        AiResponseFormat::Json => Some(OpenAiResponseFormat {
            format_type: "json_object".to_string(),
        }),
        AiResponseFormat::Text => None,
    };

    let oai_request = OpenAiChatRequest {
        model: model_name.to_string(),
        messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        response_format,
    };

    debug!("Sending chat request to LM Studio model '{}'", model_name);

    let resp = http
        .post(&url)
        .json(&oai_request)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                AiError::Timeout(format!("LM Studio request timed out: {}", e))
            } else if e.is_connect() {
                AiError::LocalServerOffline(format!(
                    "Cannot reach LM Studio at {}: {}",
                    base_url, e
                ))
            } else {
                AiError::NetworkError(e.to_string())
            }
        })?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(AiError::ApiError { status, body });
    }

    let parsed: OpenAiChatResponse = resp.json().await.map_err(|e| {
        AiError::ParseError(format!("Failed to parse LM Studio response: {}", e))
    })?;

    let content = parsed
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(AiChatResponse {
        content,
        model_used: parsed.model.unwrap_or_else(|| model_name.to_string()),
        provider: AiProvider::LmStudio,
        usage: parsed.usage.map(|u| AiUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }),
    })
}

/// List models available on LM Studio.
pub async fn list_models(
    http: &Client,
    config: &AiConfig,
) -> Result<Vec<LmStudioModel>, AiError> {
    let base_url = config
        .lm_studio_url
        .as_ref()
        .ok_or_else(|| AiError::ProviderNotConfigured("LM_STUDIO_URL not set".to_string()))?;

    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));

    let resp = http
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() || e.is_connect() {
                AiError::LocalServerOffline(format!(
                    "Cannot reach LM Studio at {}: {}",
                    base_url, e
                ))
            } else {
                AiError::NetworkError(e.to_string())
            }
        })?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(AiError::ApiError {
            status: 0,
            body: format!("Failed to list models: {}", body),
        });
    }

    let parsed: OpenAiModelsResponse = resp
        .json()
        .await
        .map_err(|e| AiError::ParseError(format!("Failed to parse models response: {}", e)))?;

    Ok(parsed
        .data
        .into_iter()
        .map(|m| LmStudioModel {
            id: m.id,
            object: m.object,
            owned_by: m.owned_by,
        })
        .collect())
}

/// Load a specific model on LM Studio.
///
/// This tells LM Studio to load the model into GPU memory. Useful when the
/// gaming PC just booted and no model is loaded yet.
pub async fn load_model(
    http: &Client,
    config: &AiConfig,
    model_name: &str,
) -> Result<(), AiError> {
    let base_url = config
        .lm_studio_url
        .as_ref()
        .ok_or_else(|| AiError::ProviderNotConfigured("LM_STUDIO_URL not set".to_string()))?;

    let url = format!("{}/api/v1/models/load", base_url.trim_end_matches('/'));

    info!(
        "Loading model '{}' on LM Studio at {}",
        model_name, base_url
    );

    let resp = http
        .post(&url)
        .json(&LmStudioLoadRequest {
            model: model_name.to_string(),
        })
        .timeout(Duration::from_secs(300)) // model loading can take a while
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() || e.is_connect() {
                AiError::LocalServerOffline(format!(
                    "Cannot reach LM Studio at {}: {}",
                    base_url, e
                ))
            } else {
                AiError::NetworkError(e.to_string())
            }
        })?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(AiError::ModelLoadFailed(format!(
            "HTTP {}: {}",
            status, body
        )));
    }

    info!("Model '{}' loaded successfully", model_name);
    Ok(())
}

/// Ensure a model is loaded in LM Studio.
/// Checks the model list first; if not loaded, triggers a load.
pub async fn ensure_model_loaded(
    http: &Client,
    config: &AiConfig,
    model_name: &str,
) -> Result<(), AiError> {
    let models = list_models(http, config).await?;

    let is_loaded = models
        .iter()
        .any(|m| m.id == model_name || m.id.contains(model_name) || model_name.contains(&m.id));

    if is_loaded {
        debug!("Model '{}' is already loaded", model_name);
        return Ok(());
    }

    info!(
        "Model '{}' not loaded — triggering auto-load. Loaded models: {:?}",
        model_name,
        models.iter().map(|m| &m.id).collect::<Vec<_>>()
    );

    load_model(http, config, model_name).await
}
