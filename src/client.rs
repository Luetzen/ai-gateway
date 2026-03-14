//! The main AI client that ties everything together.

use crate::config::AiConfig;
use crate::error::AiError;
use crate::models::*;
use crate::providers::{anthropic, gemini, lm_studio};

use reqwest::Client;
use std::time::Duration;
use tracing::{debug, warn};

/// The main AI client — routes requests to the correct provider.
///
/// Thread-safe and cheap to clone (shares the inner HTTP client via Arc).
pub struct AiClient {
    config: AiConfig,
    http: Client,
}

impl AiClient {
    /// Create a new AI client with the given configuration.
    pub fn new(config: AiConfig) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self { config, http }
    }

    /// Create a new AI client, loading configuration from environment variables.
    pub fn from_env() -> Self {
        Self::new(AiConfig::from_env())
    }

    /// Get the current configuration (read-only).
    pub fn config(&self) -> &AiConfig {
        &self.config
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /// Send a chat completion request.
    ///
    /// The model selection logic:
    /// - `AiModel::Local(name)`    → LM Studio
    /// - `AiModel::Cloud(variant)` → Anthropic
    /// - `AiModel::Gemini(variant)`→ Google Gemini
    /// - `AiModel::Auto`           → local → Anthropic → Gemini (first available)
    pub async fn chat(&self, request: AiChatRequest) -> Result<AiChatResponse, AiError> {
        match &request.model {
            AiModel::Local(model_name) => {
                lm_studio::chat_local(&self.http, &self.config, &request, model_name).await
            }
            AiModel::Cloud(variant) => {
                anthropic::chat_anthropic(&self.http, &self.config, &request, *variant).await
            }
            AiModel::Gemini(variant) => {
                gemini::chat_gemini(&self.http, &self.config, &request, *variant).await
            }
            AiModel::Auto => self.chat_auto(request).await,
        }
    }

    /// Check the status of all configured AI providers.
    pub async fn status(&self) -> AiServerStatus {
        let (lm_online, models) = if self.config.lm_studio_url.is_some() {
            match self.list_local_models().await {
                Ok(models) => (true, models.into_iter().map(|m| m.id).collect()),
                Err(_) => (false, vec![]),
            }
        } else {
            (false, vec![])
        };

        AiServerStatus {
            lm_studio_online: lm_online,
            loaded_models: models,
            anthropic_configured: self.config.anthropic_api_key.is_some(),
            gemini_configured: self.config.gemini_api_key.is_some(),
        }
    }

    /// List models available on LM Studio.
    pub async fn list_local_models(&self) -> Result<Vec<LmStudioModel>, AiError> {
        lm_studio::list_models(&self.http, &self.config).await
    }

    /// Load a specific model on LM Studio.
    ///
    /// This tells LM Studio to load the model into GPU memory. Useful when the
    /// gaming PC just booted and no model is loaded yet.
    pub async fn load_local_model(&self, model_name: &str) -> Result<(), AiError> {
        lm_studio::load_model(&self.http, &self.config, model_name).await
    }

    /// Check if LM Studio is reachable (quick health check).
    pub async fn is_local_online(&self) -> bool {
        if self.config.lm_studio_url.is_none() {
            return false;
        }
        self.list_local_models().await.is_ok()
    }

    // =========================================================================
    // Auto Mode
    // =========================================================================

    /// Auto mode: tries local first, then Anthropic, then Gemini.
    async fn chat_auto(&self, request: AiChatRequest) -> Result<AiChatResponse, AiError> {
        // --- 1. Try LM Studio (local) ---
        if let Some(ref model_name) = self.config.default_local_model {
            if self.config.lm_studio_url.is_some() {
                debug!("Auto mode: trying local model '{}'", model_name);
                match lm_studio::chat_local(&self.http, &self.config, &request, model_name).await {
                    Ok(resp) => return Ok(resp),
                    Err(e) => {
                        warn!(
                            "Auto mode: local model '{}' failed ({}), trying next provider",
                            model_name, e
                        );
                    }
                }
            }
        } else if self.config.lm_studio_url.is_some() {
            // No default model set, but LM Studio is configured — use first loaded model
            if let Ok(models) = self.list_local_models().await {
                if let Some(first) = models.first() {
                    debug!("Auto mode: using first loaded local model '{}'", first.id);
                    match lm_studio::chat_local(&self.http, &self.config, &request, &first.id).await
                    {
                        Ok(resp) => return Ok(resp),
                        Err(e) => {
                            warn!(
                                "Auto mode: local model '{}' failed ({}), trying next provider",
                                first.id, e
                            );
                        }
                    }
                }
            }
        }

        // --- 2. Try Anthropic ---
        if self.config.anthropic_api_key.is_some() {
            debug!(
                "Auto mode: trying Anthropic model '{}'",
                self.config.default_cloud_model
            );
            match anthropic::chat_anthropic(
                &self.http,
                &self.config,
                &request,
                self.config.default_cloud_model,
            )
            .await
            {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    warn!("Auto mode: Anthropic failed ({}), trying Gemini", e);
                }
            }
        }

        // --- 3. Try Gemini ---
        if self.config.gemini_api_key.is_some() {
            debug!(
                "Auto mode: trying Gemini model '{}'",
                self.config.default_gemini_model
            );
            return gemini::chat_gemini(
                &self.http,
                &self.config,
                &request,
                self.config.default_gemini_model,
            )
            .await;
        }

        Err(AiError::NoProviderAvailable(
            "No AI provider configured. \
             Set LM_STUDIO_URL, ANTHROPIC_API_KEY, or GEMINI_API_KEY."
                .to_string(),
        ))
    }
}
