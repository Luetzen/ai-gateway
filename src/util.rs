//! Utility functions for the AI gateway.

use crate::error::AiError;
use std::time::Duration;

/// Exponential backoff: 5s, 15s, 45s, capped at 90s.
pub(crate) fn backoff_duration(attempt: u32) -> Duration {
    let secs = (5u64 * 3u64.pow(attempt.saturating_sub(1))).min(90);
    Duration::from_secs(secs)
}

/// Strip markdown code fences from AI output.
///
/// Many models wrap JSON in \`\`\`json ... \`\`\` blocks. This utility
/// strips those fences to get clean content.
pub fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();
    if let Some(stripped) = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
    {
        stripped.trim().trim_end_matches("```").trim()
    } else {
        trimmed
    }
}

/// Parse a JSON response from AI output, automatically stripping code fences.
pub fn parse_ai_json<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, AiError> {
    let clean = strip_code_fences(text);
    serde_json::from_str(clean).map_err(|e| {
        AiError::ParseError(format!(
            "Failed to parse AI JSON: {}. Raw text: {}",
            e,
            &text[..text.len().min(500)]
        ))
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn test_strip_code_fences_json() {
        let input = "```json\n{\"key\": \"value\"}\n```";
        assert_eq!(strip_code_fences(input), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_strip_code_fences_plain() {
        let input = "```\n{\"key\": \"value\"}\n```";
        assert_eq!(strip_code_fences(input), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_strip_code_fences_none() {
        let input = "{\"key\": \"value\"}";
        assert_eq!(strip_code_fences(input), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_parse_ai_json() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct TestData {
            name: String,
            count: i32,
        }

        let input = "```json\n{\"name\": \"test\", \"count\": 42}\n```";
        let parsed: TestData = parse_ai_json(input).expect("should parse");
        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.count, 42);
    }

    #[test]
    fn test_backoff_duration() {
        assert_eq!(backoff_duration(1), Duration::from_secs(5));
        assert_eq!(backoff_duration(2), Duration::from_secs(15));
        assert_eq!(backoff_duration(3), Duration::from_secs(45));
        assert_eq!(backoff_duration(4), Duration::from_secs(90)); // capped
        assert_eq!(backoff_duration(5), Duration::from_secs(90)); // still capped
    }
}
