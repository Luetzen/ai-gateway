//! Public model types for the AI gateway.

use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Anthropic Model Variants
// =============================================================================

/// Anthropic model variants — always resolves to the latest version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicModel {
    Haiku,
    Sonnet,
    Opus,
}

impl AnthropicModel {
    /// Returns the latest model identifier string for the Anthropic API.
    pub fn model_id(&self) -> &'static str {
        match self {
            AnthropicModel::Haiku => "claude-haiku-4-5-20251001",
            AnthropicModel::Sonnet => "claude-sonnet-4-20250514",
            AnthropicModel::Opus => "claude-opus-4-20250514",
        }
    }
}

impl fmt::Display for AnthropicModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnthropicModel::Haiku => write!(f, "haiku"),
            AnthropicModel::Sonnet => write!(f, "sonnet"),
            AnthropicModel::Opus => write!(f, "opus"),
        }
    }
}

// =============================================================================
// Gemini Model Variants
// =============================================================================

/// Google Gemini model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeminiModel {
    /// Gemini 2.5 Flash — fast, free tier available, multimodal.
    Flash,
    /// Gemini 2.5 Pro — most capable Gemini model.
    Pro,
    /// Gemini 2.0 Flash Lite — ultra-fast, very cheap, free tier.
    FlashLite,
}

impl GeminiModel {
    /// Returns the model identifier string for the Gemini API.
    pub fn model_id(&self) -> &'static str {
        match self {
            GeminiModel::Flash => "gemini-2.5-flash",
            GeminiModel::Pro => "gemini-2.5-pro",
            GeminiModel::FlashLite => "gemini-2.0-flash-lite",
        }
    }
}

impl fmt::Display for GeminiModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeminiModel::Flash => write!(f, "flash"),
            GeminiModel::Pro => write!(f, "pro"),
            GeminiModel::FlashLite => write!(f, "flash_lite"),
        }
    }
}

// =============================================================================
// Model Selection
// =============================================================================

/// Which AI model to use for a request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum AiModel {
    /// A local model served by LM Studio.
    /// The string is the model identifier as known by LM Studio
    /// (e.g. "qwen3-30b-a3b", "gemma-3-12b-it").
    Local(String),

    /// An Anthropic cloud model.
    Cloud(AnthropicModel),

    /// A Google Gemini cloud model.
    Gemini(GeminiModel),

    /// Use the best available: tries local first, falls back to cloud.
    #[default]
    Auto,
}

// =============================================================================
// Messages
// =============================================================================

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiMessage {
    pub role: AiRole,
    pub content: String,
}

impl AiMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: AiRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: AiRole::Assistant,
            content: content.into(),
        }
    }
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AiRole {
    User,
    Assistant,
}

impl fmt::Display for AiRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiRole::User => write!(f, "user"),
            AiRole::Assistant => write!(f, "assistant"),
        }
    }
}

// =============================================================================
// Request / Response
// =============================================================================

/// Desired response format.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiResponseFormat {
    /// Free-form text (default).
    #[default]
    Text,
    /// Request JSON output. The AI will try to produce valid JSON.
    Json,
}

/// A chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiChatRequest {
    /// Which model to use.
    #[serde(default)]
    pub model: AiModel,

    /// System prompt (sets the AI's behavior/persona).
    pub system: Option<String>,

    /// Conversation messages.
    pub messages: Vec<AiMessage>,

    /// Maximum tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Sampling temperature (0.0 = deterministic, 1.0+ = creative).
    pub temperature: Option<f64>,

    /// Desired response format.
    #[serde(default)]
    pub response_format: AiResponseFormat,
}

fn default_max_tokens() -> u32 {
    4096
}

/// A chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiChatResponse {
    /// The generated text content.
    pub content: String,

    /// Which model actually processed the request.
    pub model_used: String,

    /// Which provider handled the request.
    pub provider: AiProvider,

    /// Token usage statistics (if available).
    pub usage: Option<AiUsage>,
}

/// Which provider handled the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AiProvider {
    LmStudio,
    Anthropic,
    Gemini,
}

impl fmt::Display for AiProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiProvider::LmStudio => write!(f, "lm_studio"),
            AiProvider::Anthropic => write!(f, "anthropic"),
            AiProvider::Gemini => write!(f, "gemini"),
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// Information about a model available on LM Studio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmStudioModel {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub owned_by: String,
}

/// Status of the local AI server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiServerStatus {
    /// Whether the LM Studio server is reachable.
    pub lm_studio_online: bool,

    /// List of loaded/available models (empty if offline).
    pub loaded_models: Vec<String>,

    /// Whether Anthropic API key is configured.
    pub anthropic_configured: bool,

    /// Whether Google Gemini API key is configured.
    pub gemini_configured: bool,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_model_ids() {
        assert!(AnthropicModel::Haiku.model_id().contains("haiku"));
        assert!(AnthropicModel::Sonnet.model_id().contains("sonnet"));
        assert!(AnthropicModel::Opus.model_id().contains("opus"));
    }

    #[test]
    fn test_ai_message_constructors() {
        let user_msg = AiMessage::user("Hello");
        assert_eq!(user_msg.role, AiRole::User);
        assert_eq!(user_msg.content, "Hello");

        let asst_msg = AiMessage::assistant("Hi there");
        assert_eq!(asst_msg.role, AiRole::Assistant);
        assert_eq!(asst_msg.content, "Hi there");
    }

    #[test]
    fn test_ai_model_default() {
        match AiModel::default() {
            AiModel::Auto => {} // expected
            other => panic!("Expected Auto, got {:?}", other),
        }
    }

    #[test]
    fn test_ai_response_format_default() {
        match AiResponseFormat::default() {
            AiResponseFormat::Text => {} // expected
            other => panic!("Expected Text, got {:?}", other),
        }
    }
}
