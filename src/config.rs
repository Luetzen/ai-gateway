//! Configuration for the AI client.

use crate::models::{AnthropicModel, GeminiModel};

/// Configuration for the AI client.
///
/// Designed to be loaded from environment variables so it works identically
/// in Home App, Pixel, or any other project.
#[derive(Debug, Clone)]
pub struct AiConfig {
    /// Base URL for LM Studio (e.g. "http://192.168.1.121:1234")
    /// None = local AI disabled.
    pub lm_studio_url: Option<String>,

    /// Anthropic API key for cloud models (Haiku, Sonnet, Opus).
    /// None = cloud AI disabled.
    pub anthropic_api_key: Option<String>,

    /// Google Gemini API key for Gemini cloud models.
    /// None = Gemini disabled.
    /// Free tier available at https://aistudio.google.com/apikey
    pub gemini_api_key: Option<String>,

    /// Default local model identifier as known by LM Studio.
    /// e.g. "qwen3-30b-a3b" or "gemma-3-12b-it"
    pub default_local_model: Option<String>,

    /// Default cloud model to use when no specific model is requested.
    pub default_cloud_model: AnthropicModel,

    /// Default Gemini model to use when no specific Gemini model is requested.
    pub default_gemini_model: GeminiModel,

    /// Whether to auto-load models in LM Studio before sending a request.
    pub auto_load_models: bool,

    /// Request timeout in seconds.
    pub request_timeout_secs: u64,

    /// Maximum retry attempts for transient errors.
    pub max_retries: u32,
}

impl AiConfig {
    /// Load configuration from environment variables.
    ///
    /// | Env Variable | Description |
    /// |---|---|
    /// | `LM_STUDIO_URL` | Base URL for LM Studio |
    /// | `ANTHROPIC_API_KEY` | Anthropic API key |
    /// | `GEMINI_API_KEY` | Google Gemini API key (free tier available) |
    /// | `AI_DEFAULT_LOCAL_MODEL` | Default local model name |
    /// | `AI_DEFAULT_CLOUD_MODEL` | "haiku", "sonnet", or "opus" |
    /// | `AI_DEFAULT_GEMINI_MODEL` | "flash", "pro", or "flash_lite" (default: flash) |
    /// | `AI_AUTO_LOAD_MODELS` | "true"/"false" (default: true) |
    /// | `AI_REQUEST_TIMEOUT_SECS` | Timeout in seconds (default: 120) |
    /// | `AI_MAX_RETRIES` | Max retries (default: 3) |
    pub fn from_env() -> Self {
        let default_cloud = match std::env::var("AI_DEFAULT_CLOUD_MODEL")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "opus" => AnthropicModel::Opus,
            "haiku" => AnthropicModel::Haiku,
            _ => AnthropicModel::Sonnet,
        };

        let default_gemini = match std::env::var("AI_DEFAULT_GEMINI_MODEL")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "pro" => GeminiModel::Pro,
            "flash_lite" | "flashlite" => GeminiModel::FlashLite,
            _ => GeminiModel::Flash,
        };

        Self {
            lm_studio_url: std::env::var("LM_STUDIO_URL")
                .ok()
                .filter(|s| !s.is_empty()),
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            gemini_api_key: std::env::var("GEMINI_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            default_local_model: std::env::var("AI_DEFAULT_LOCAL_MODEL")
                .ok()
                .filter(|s| !s.is_empty()),
            default_cloud_model: default_cloud,
            default_gemini_model: default_gemini,
            auto_load_models: std::env::var("AI_AUTO_LOAD_MODELS")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            request_timeout_secs: std::env::var("AI_REQUEST_TIMEOUT_SECS")
                .unwrap_or_else(|_| "120".to_string())
                .parse()
                .unwrap_or(120),
            max_retries: std::env::var("AI_MAX_RETRIES")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .unwrap_or(3),
        }
    }

    /// Create a config with explicit values (useful for testing or non-env setups).
    pub fn new(lm_studio_url: Option<String>, anthropic_api_key: Option<String>) -> Self {
        Self {
            lm_studio_url,
            anthropic_api_key,
            gemini_api_key: None,
            default_local_model: None,
            default_cloud_model: AnthropicModel::Sonnet,
            default_gemini_model: GeminiModel::Flash,
            auto_load_models: true,
            request_timeout_secs: 120,
            max_retries: 3,
        }
    }

    /// Check if any AI provider is configured.
    pub fn is_any_provider_available(&self) -> bool {
        self.lm_studio_url.is_some()
            || self.anthropic_api_key.is_some()
            || self.gemini_api_key.is_some()
    }

    /// Check if local AI (LM Studio) is configured.
    pub fn is_local_available(&self) -> bool {
        self.lm_studio_url.is_some()
    }

    /// Check if cloud AI (Anthropic) is configured.
    pub fn is_cloud_available(&self) -> bool {
        self.anthropic_api_key.is_some()
    }

    /// Check if Google Gemini is configured.
    pub fn is_gemini_available(&self) -> bool {
        self.gemini_api_key.is_some()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = AiConfig::new(None, None);
        assert!(!config.is_any_provider_available());
        assert!(!config.is_local_available());
        assert!(!config.is_cloud_available());
        assert!(config.auto_load_models);
        assert_eq!(config.request_timeout_secs, 120);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_config_with_providers() {
        let config = AiConfig::new(
            Some("http://192.168.1.121:1234".to_string()),
            Some("sk-ant-test".to_string()),
        );
        assert!(config.is_any_provider_available());
        assert!(config.is_local_available());
        assert!(config.is_cloud_available());
    }
}
