//! # ai-gateway
//!
//! Provider-agnostic AI gateway for Rust applications.
//!
//! Supports:
//! - **LM Studio** (local, OpenAI-compatible API)
//! - **Anthropic** (cloud — Haiku, Sonnet, Opus)
//! - **Google Gemini** (cloud — Flash, Pro, Flash Lite — free tier available)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ai_gateway::{AiClient, AiChatRequest, AiMessage, AiModel};
//!
//! // Anthropic
//! let client = AiClient::from_env();
//! let response = client.chat(AiChatRequest {
//!     model: AiModel::Cloud(AnthropicModel::Sonnet),
//!     system: Some("You are a helpful assistant.".into()),
//!     messages: vec![AiMessage::user("Hello!")],
//!     max_tokens: 2048,
//!     temperature: Some(0.7),
//!     response_format: Default::default(),
//! }).await?;
//!
//! // Google Gemini (free tier via GEMINI_API_KEY)
//! let response = client.chat(AiChatRequest {
//!     model: AiModel::Gemini(GeminiModel::Flash),
//!     system: Some("You are a helpful assistant.".into()),
//!     messages: vec![AiMessage::user("Hello!")],
//!     max_tokens: 2048,
//!     temperature: Some(0.7),
//!     response_format: Default::default(),
//! }).await?;
//!
//! // Auto: tries local → Anthropic → Gemini
//! let response = client.chat(AiChatRequest {
//!     model: AiModel::Auto,
//!     system: Some("You are a helpful assistant.".into()),
//!     messages: vec![AiMessage::user("Hello!")],
//!     max_tokens: 2048,
//!     temperature: Some(0.7),
//!     response_format: Default::default(),
//! }).await?;
//! ```

mod client;
mod config;
mod error;
mod models;
pub mod providers;
mod util;

// Re-export everything at crate root for ergonomic usage
pub use client::AiClient;
pub use config::AiConfig;
pub use error::AiError;
pub use models::*;
pub use util::{parse_ai_json, strip_code_fences};
