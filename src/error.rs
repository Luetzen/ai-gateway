//! Error types for the AI gateway.

use std::fmt;

/// AI client errors.
#[derive(Debug)]
pub enum AiError {
    /// The requested provider is not configured.
    ProviderNotConfigured(String),

    /// LM Studio server is not reachable (gaming PC off?).
    LocalServerOffline(String),

    /// Model could not be loaded on LM Studio.
    ModelLoadFailed(String),

    /// The API returned a non-success status.
    ApiError { status: u16, body: String },

    /// Rate limited (429) — includes retry-after seconds if available.
    RateLimited {
        retry_after_secs: Option<u64>,
        body: String,
    },

    /// Server overloaded (529/503).
    Overloaded(String),

    /// Network / connection error.
    NetworkError(String),

    /// Failed to parse the response.
    ParseError(String),

    /// No provider available for the requested model type.
    NoProviderAvailable(String),

    /// Request timed out.
    Timeout(String),
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiError::ProviderNotConfigured(msg) => write!(f, "Provider not configured: {}", msg),
            AiError::LocalServerOffline(msg) => write!(f, "Local AI server offline: {}", msg),
            AiError::ModelLoadFailed(msg) => write!(f, "Model load failed: {}", msg),
            AiError::ApiError { status, body } => {
                write!(f, "AI API error (HTTP {}): {}", status, body)
            }
            AiError::RateLimited { body, .. } => write!(f, "Rate limited: {}", body),
            AiError::Overloaded(msg) => write!(f, "Server overloaded: {}", msg),
            AiError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            AiError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            AiError::NoProviderAvailable(msg) => write!(f, "No AI provider available: {}", msg),
            AiError::Timeout(msg) => write!(f, "Request timed out: {}", msg),
        }
    }
}

impl std::error::Error for AiError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_error_display() {
        let err = AiError::LocalServerOffline("PC is off".to_string());
        assert!(err.to_string().contains("PC is off"));

        let err = AiError::RateLimited {
            retry_after_secs: Some(30),
            body: "slow down".to_string(),
        };
        assert!(err.to_string().contains("slow down"));
    }
}
