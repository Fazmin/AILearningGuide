use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tauri::{AppHandle, Emitter};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderTestRequest {
    pub provider: String,
    pub secret: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ProviderTestResult {
    pub ok: bool,
    pub message: String,
    pub model: Option<String>,
    pub models: Vec<String>,
}

pub fn compatible_models_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        String::new()
    } else if trimmed.ends_with("/models") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/models")
    }
}

pub fn redact_secret(text: &str, secret: &str) -> String {
    if secret.is_empty() {
        text.to_string()
    } else {
        text.replace(secret, "•••")
    }
}

fn keep_chat_model(id: &str) -> bool {
    let lower = id.to_ascii_lowercase();
    let skipped = [
        "embedding",
        "whisper",
        "tts",
        "audio",
        "realtime",
        "transcribe",
        "image",
        "dall-e",
        "davinci",
        "babbage",
        "moderation",
        "search",
        "omni",
    ];
    !id.trim().is_empty() && !skipped.iter().any(|needle| lower.contains(needle))
}

fn parse_models(payload: &serde_json::Value) -> Vec<String> {
    let rows = payload
        .get("data")
        .or_else(|| payload.get("models"))
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();

    rows.into_iter()
        .filter_map(|row| {
            let raw = row
                .get("id")
                .or_else(|| row.get("name"))
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let id = raw.trim_start_matches("models/").trim();
            keep_chat_model(id).then(|| id.to_string())
        })
        .take(40)
        .collect()
}

fn interpret(status: u16, models: Vec<String>, selected: Option<String>) -> ProviderTestResult {
    let model = selected.filter(|value| !value.trim().is_empty());
    if status == 401 || status == 403 {
        return ProviderTestResult {
            ok: false,
            message: "The provider rejected this key.".into(),
            model,
            models: Vec::new(),
        };
    }
    if status == 429 {
        return ProviderTestResult {
            ok: true,
            message: "The key was accepted, but the provider is rate-limiting right now.".into(),
            model,
            models,
        };
    }
    if status == 404 {
        return ProviderTestResult {
            ok: false,
            message: "The models endpoint was not found. Check the base URL.".into(),
            model,
            models: Vec::new(),
        };
    }
    if !(200..300).contains(&status) {
        return ProviderTestResult {
            ok: false,
            message: format!("The provider returned HTTP {status}."),
            model,
            models: Vec::new(),
        };
    }
    if let Some(name) = model.as_deref() {
        if !models.is_empty() && !models.iter().any(|item| item == name) {
            return ProviderTestResult {
                ok: true,
                message: format!("The key works, but this account cannot see {name}. Choose another model."),
                model,
                models,
            };
        }
        return ProviderTestResult {
            ok: true,
            message: format!("Connection ready · {name}"),
            model,
            models,
        };
    }
    ProviderTestResult {
        ok: true,
        message: "Connection ready. The key can list models.".into(),
        model,
        models,
    }
}

pub async fn test_connection(request: ProviderTestRequest) -> Result<ProviderTestResult, String> {
    let secret = request.secret.as_deref().unwrap_or("").trim();
    if secret.is_empty() {
        return Ok(ProviderTestResult {
            ok: false,
            message: "Enter a key, or store one first.".into(),
            model: None,
            models: Vec::new(),
        });
    }
    if request.provider == "compatible" && request.base_url.as_deref().unwrap_or("").trim().is_empty()
    {
        return Ok(ProviderTestResult {
            ok: false,
            message: "Enter the OpenAI-compatible base URL.".into(),
            model: None,
            models: Vec::new(),
        });
    }

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(8))
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|error| error.to_string())?;

    let request_builder = match request.provider.as_str() {
        "openai" => client
            .get("https://api.openai.com/v1/models")
            .bearer_auth(secret),
        "anthropic" => client
            .get("https://api.anthropic.com/v1/models")
            .header("x-api-key", secret)
            .header("anthropic-version", "2023-06-01"),
        "google" => client
            .get("https://generativelanguage.googleapis.com/v1beta/models")
            .header("x-goog-api-key", secret),
        "compatible" => client
            .get(compatible_models_url(request.base_url.as_deref().unwrap_or("")))
            .bearer_auth(secret),
        other => {
            return Ok(ProviderTestResult {
                ok: false,
                message: format!("Unknown provider: {other}"),
                model: None,
                models: Vec::new(),
            });
        }
    };

    let response = request_builder.send().await.map_err(|error| redact_secret(&error.to_string(), secret))?;
    let status = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    let payload = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
    Ok(interpret(
        status,
        parse_models(&payload),
        request.model.filter(|value| !value.trim().is_empty()),
    ))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CloudChatTurn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CloudChatRequest {
    pub request_id: String,
    pub provider: String,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub module_title: String,
    pub mode: String,
    pub objectives: Vec<String>,
    pub current_step: String,
    pub state_json: String,
    pub message: String,
    pub history: Option<Vec<CloudChatTurn>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GuideChatChunk {
    request_id: String,
    content: String,
    done: bool,
}

pub fn default_model(provider: &str) -> &'static str {
    match provider {
        "openai" => "gpt-5.6-terra",
        "anthropic" => "claude-sonnet-5",
        "google" => "gemini-3.8-flash",
        _ => "",
    }
}

pub fn compatible_chat_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        String::new()
    } else if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

/// How the topic guide pitches its answers for the learner's explanation mode. Plain matches
/// the Plain lessons, which are written at a grade 8 to 10 reading level (CONTENT_STYLE_GUIDE.md).
pub fn explanation_style(mode: &str) -> &'static str {
    if mode == "plain" {
        "Explain at a grade 8 to 10 reading level: everyday words, describe any technical term before using it, and give a concrete example or a short everyday comparison when it helps."
    } else {
        "Explain in standard language: use the accepted technical vocabulary and define specialized terms near first use."
    }
}

pub fn topic_guide_system_prompt(
    module_title: &str,
    mode: &str,
    objectives: &[String],
    current_step: &str,
    state_json: &str,
) -> String {
    let objectives = objectives
        .iter()
        .map(|objective| format!("- {objective}"))
        .collect::<Vec<_>>()
        .join("\n");
    let style = explanation_style(mode);
    format!(
        "You are the topic guide for the learning module \"{module_title}\".\n\
         Stay within this module. If the learner asks about another subject, briefly redirect them.\n\
         {style} Be concise, accurate, and distinguish the toy visualization from real models.\n\
         Learning objectives:\n{objectives}\n\
         Current step: {current_step}\n\
         Current live interactive state: {state_json}\n\
         Refer to the live state when it helps. Do not reveal hidden chain-of-thought; provide only the useful answer."
    )
}

fn json_text(value: &serde_json::Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return (!text.is_empty()).then(|| text.to_string());
    }
    if let Some(parts) = value.as_array() {
        let text = parts
            .iter()
            .filter_map(|part| {
                part.as_str()
                    .or_else(|| part.get("text").and_then(|item| item.as_str()))
                    .or_else(|| part.get("content").and_then(|item| item.as_str()))
            })
            .collect::<String>();
        return (!text.is_empty()).then_some(text);
    }
    None
}

pub fn extract_stream_text(value: &serde_json::Value) -> Option<String> {
    value
        .pointer("/choices/0/delta/content")
        .and_then(json_text)
        .or_else(|| value.pointer("/delta/text").and_then(json_text))
        .or_else(|| value.get("delta").and_then(json_text))
        .or_else(|| value.pointer("/candidates/0/content/parts/0/text").and_then(json_text))
}

pub fn extract_complete_text(value: &serde_json::Value) -> Option<String> {
    extract_stream_text(value)
        .or_else(|| value.pointer("/choices/0/message/content").and_then(json_text))
        .or_else(|| value.pointer("/output_text").and_then(json_text))
        .or_else(|| value.pointer("/response/output_text").and_then(json_text))
}

fn emit_chunk(app: &AppHandle, request_id: &str, content: &str, done: bool) {
    let _ = app.emit(
        "local-chat-chunk",
        GuideChatChunk {
            request_id: request_id.to_string(),
            content: content.to_string(),
            done,
        },
    );
}

fn provider_error(status: u16, provider: &str) -> String {
    match status {
        401 | 403 => format!(
            "The stored {provider} key was rejected. Update it in Settings → Cloud providers."
        ),
        429 => "The provider is rate-limiting this key. Try again in a moment.".into(),
        _ => format!("The provider returned HTTP {status}."),
    }
}

async fn read_sse_text(
    app: &AppHandle,
    request_id: &str,
    response: reqwest::Response,
    secret: &str,
) -> Result<String, String> {
    let mut stream = response.bytes_stream();
    let mut pending = Vec::<u8>::new();
    let mut answer = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| redact_secret(&error.to_string(), secret))?;
        pending.extend_from_slice(&chunk);
        while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
            let line = pending.drain(..=newline).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line);
            let data = line.trim().strip_prefix("data:").map(str::trim);
            let Some(data) = data else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            if let Some(content) = extract_stream_text(&value) {
                answer.push_str(&content);
                emit_chunk(app, request_id, &content, false);
            } else if answer.is_empty() {
                if let Some(content) = extract_complete_text(&value) {
                    answer.push_str(&content);
                    emit_chunk(app, request_id, &content, false);
                }
            }
        }
    }
    if answer.is_empty() {
        let leftover = String::from_utf8_lossy(&pending);
        let data = leftover.trim().strip_prefix("data:").map(str::trim).unwrap_or(leftover.trim());
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(content) = extract_complete_text(&value) {
                emit_chunk(app, request_id, &content, false);
                return Ok(content);
            }
        }
    }
    Ok(answer)
}

fn openai_messages(system: &str, history: &[CloudChatTurn], message: &str) -> serde_json::Value {
    let mut messages = vec![serde_json::json!({ "role": "system", "content": system })];
    for turn in history {
        let role = if turn.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        if !turn.content.trim().is_empty() {
            messages.push(serde_json::json!({ "role": role, "content": turn.content }));
        }
    }
    messages.push(serde_json::json!({ "role": "user", "content": message }));
    serde_json::Value::Array(messages)
}

async fn complete_openai_compatible(
    app: &AppHandle,
    request_id: &str,
    client: &Client,
    url: &str,
    secret: &str,
    model: &str,
    system: &str,
    history: &[CloudChatTurn],
    message: &str,
) -> Result<String, String> {
    let payload = serde_json::json!({
        "model": model,
        "messages": openai_messages(system, history, message),
        "stream": true,
        "max_completion_tokens": 800
    });
    let response = client
        .post(url)
        .bearer_auth(secret)
        .json(&payload)
        .send()
        .await
        .map_err(|error| redact_secret(&error.to_string(), secret))?;
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = redact_secret(&response.text().await.unwrap_or_default(), secret);
        if url.contains("api.openai.com") && (status == 400 || status == 404) {
            return complete_openai_responses(
                app, request_id, client, secret, model, system, history, message,
            )
            .await
            .map_err(|error| {
                if error.contains("HTTP") {
                    error
                } else {
                    format!("{error} First attempt: HTTP {status} {}", body.chars().take(160).collect::<String>())
                }
            });
        }
        return Err(provider_error(status, "OpenAI"));
    }
    let answer = read_sse_text(app, request_id, response, secret).await?;
    if answer.trim().is_empty() && url.contains("api.openai.com") {
        return complete_openai_responses(
            app, request_id, client, secret, model, system, history, message,
        )
        .await;
    }
    Ok(answer)
}

async fn complete_openai_responses(
    app: &AppHandle,
    request_id: &str,
    client: &Client,
    secret: &str,
    model: &str,
    system: &str,
    history: &[CloudChatTurn],
    message: &str,
) -> Result<String, String> {
    let mut input = Vec::new();
    for turn in history {
        let role = if turn.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        if !turn.content.trim().is_empty() {
            input.push(serde_json::json!({ "role": role, "content": turn.content }));
        }
    }
    input.push(serde_json::json!({ "role": "user", "content": message }));
    let response = client
        .post("https://api.openai.com/v1/responses")
        .bearer_auth(secret)
        .json(&serde_json::json!({
            "model": model,
            "instructions": system,
            "input": input,
            "stream": true
        }))
        .send()
        .await
        .map_err(|error| redact_secret(&error.to_string(), secret))?;
    if !response.status().is_success() {
        return Err(provider_error(response.status().as_u16(), "OpenAI"));
    }
    read_sse_text(app, request_id, response, secret).await
}

async fn complete_anthropic(
    app: &AppHandle,
    request_id: &str,
    client: &Client,
    secret: &str,
    model: &str,
    system: &str,
    history: &[CloudChatTurn],
    message: &str,
) -> Result<String, String> {
    let mut messages = Vec::new();
    for turn in history {
        let role = if turn.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        if !turn.content.trim().is_empty() {
            messages.push(serde_json::json!({ "role": role, "content": turn.content }));
        }
    }
    messages.push(serde_json::json!({ "role": "user", "content": message }));
    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", secret)
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": model,
            "max_tokens": 800,
            "system": system,
            "messages": messages,
            "stream": true
        }))
        .send()
        .await
        .map_err(|error| redact_secret(&error.to_string(), secret))?;
    if !response.status().is_success() {
        return Err(provider_error(response.status().as_u16(), "Anthropic"));
    }
    read_sse_text(app, request_id, response, secret).await
}

async fn complete_google(
    app: &AppHandle,
    request_id: &str,
    client: &Client,
    secret: &str,
    model: &str,
    system: &str,
    history: &[CloudChatTurn],
    message: &str,
) -> Result<String, String> {
    let mut contents = Vec::new();
    for turn in history {
        let role = if turn.role == "assistant" { "model" } else { "user" };
        if !turn.content.trim().is_empty() {
            contents.push(serde_json::json!({
                "role": role,
                "parts": [{ "text": turn.content }]
            }));
        }
    }
    contents.push(serde_json::json!({
        "role": "user",
        "parts": [{ "text": message }]
    }));
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    );
    let response = client
        .post(url)
        .header("x-goog-api-key", secret)
        .json(&serde_json::json!({
            "systemInstruction": { "parts": [{ "text": system }] },
            "contents": contents,
            "generationConfig": { "maxOutputTokens": 800 }
        }))
        .send()
        .await
        .map_err(|error| redact_secret(&error.to_string(), secret))?;
    if !response.status().is_success() {
        return Err(provider_error(response.status().as_u16(), "Google"));
    }
    let payload = response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| error.to_string())?;
    let answer = payload
        .pointer("/candidates/0/content/parts/0/text")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    if !answer.is_empty() {
        emit_chunk(app, request_id, &answer, false);
    }
    Ok(answer)
}

pub async fn chat(app: AppHandle, request: CloudChatRequest, secret: String) -> Result<String, String> {
    let secret = secret.trim().to_string();
    if secret.is_empty() {
        return Err("No key is stored for this provider.".into());
    }
    if request.provider == "compatible" && request.base_url.as_deref().unwrap_or("").trim().is_empty()
    {
        return Err("Enter the OpenAI-compatible base URL in Settings.".into());
    }
    let model = request
        .model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default_model(&request.provider))
        .to_string();
    if model.is_empty() {
        return Err("Choose a model in Settings → Cloud providers.".into());
    }
    let system = topic_guide_system_prompt(
        &request.module_title,
        &request.mode,
        &request.objectives,
        &request.current_step,
        &request.state_json,
    );
    let history = request.history.unwrap_or_default();
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(90))
        .build()
        .map_err(|error| error.to_string())?;
    let answer = match request.provider.as_str() {
        "openai" => {
            complete_openai_compatible(
                &app,
                &request.request_id,
                &client,
                "https://api.openai.com/v1/chat/completions",
                &secret,
                &model,
                &system,
                &history,
                &request.message,
            )
            .await
        }
        "compatible" => {
            complete_openai_compatible(
                &app,
                &request.request_id,
                &client,
                &compatible_chat_url(request.base_url.as_deref().unwrap_or("")),
                &secret,
                &model,
                &system,
                &history,
                &request.message,
            )
            .await
        }
        "anthropic" => {
            complete_anthropic(
                &app,
                &request.request_id,
                &client,
                &secret,
                &model,
                &system,
                &history,
                &request.message,
            )
            .await
        }
        "google" => {
            complete_google(
                &app,
                &request.request_id,
                &client,
                &secret,
                &model,
                &system,
                &history,
                &request.message,
            )
            .await
        }
        other => Err(format!("Unknown provider: {other}")),
    }?;
    emit_chunk(&app, &request.request_id, "", true);
    if answer.trim().is_empty() {
        return Err("The provider returned an empty answer.".into());
    }
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_url_normalizes_trailing_slash() {
        assert_eq!(
            compatible_models_url("https://api.together.xyz/v1/"),
            "https://api.together.xyz/v1/models"
        );
        assert_eq!(
            compatible_models_url("http://127.0.0.1:11434/v1/models"),
            "http://127.0.0.1:11434/v1/models"
        );
    }

    #[test]
    fn redacts_secrets_from_transport_errors() {
        assert_eq!(
            redact_secret("Authorization Bearer sk-live-123 failed", "sk-live-123"),
            "Authorization Bearer ••• failed"
        );
    }

    #[test]
    fn accepted_key_with_missing_model_stays_successful() {
        let result = interpret(
            200,
            vec!["gpt-5.6-luna".into()],
            Some("gpt-5.6-terra".into()),
        );
        assert!(result.ok);
        assert!(result.message.contains("cannot see"));
    }

    #[test]
    fn chat_url_appends_completions() {
        assert_eq!(
            compatible_chat_url("https://api.together.xyz/v1/"),
            "https://api.together.xyz/v1/chat/completions"
        );
    }

    #[test]
    fn system_prompt_names_the_module_and_live_state() {
        let prompt = topic_guide_system_prompt(
            "Stacking neurons",
            "plain",
            &["Relate depth to capacity".into()],
            "Add depth",
            "{\"layers\":3}",
        );
        assert!(prompt.contains("Stacking neurons"));
        assert!(prompt.contains("Add depth"));
        assert!(prompt.contains("{\"layers\":3}"));
        assert!(prompt.contains("grade 8 to 10"));
    }

    #[test]
    fn standard_prompt_keeps_the_technical_register() {
        let prompt = topic_guide_system_prompt("Attention", "standard", &[], "Score", "{}");
        assert!(prompt.contains("technical vocabulary"));
        assert!(!prompt.contains("grade 8 to 10"));
    }

    #[test]
    fn extracts_openai_and_anthropic_stream_deltas() {
        let openai = serde_json::json!({"choices":[{"delta":{"content":"Hello"}}]});
        let anthropic = serde_json::json!({"type":"content_block_delta","delta":{"text":" there"}});
        let responses = serde_json::json!({"type":"response.output_text.delta","delta":"Hi"});
        let complete = serde_json::json!({"choices":[{"message":{"content":"Done"}}]});
        assert_eq!(extract_stream_text(&openai).as_deref(), Some("Hello"));
        assert_eq!(extract_stream_text(&anthropic).as_deref(), Some(" there"));
        assert_eq!(extract_stream_text(&responses).as_deref(), Some("Hi"));
        assert_eq!(extract_complete_text(&complete).as_deref(), Some("Done"));
    }
}
