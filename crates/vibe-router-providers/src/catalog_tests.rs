//! Hermetic tests for remote-catalog synchronization. The catalog feed is
//! served by the loopback mock upstream; provider base URLs inside the feed are
//! public hosts that are never actually contacted (registration only).

use std::collections::HashSet;

use vibe_router_config::schema::{AuthConfig, ModelConfig, ProviderConfig};
use vibe_router_config::AppConfig;
use vibe_router_core::{Auth, SecretRef, WireFormat};

use crate::catalog::sync_remote_catalog;
use crate::mock::{MockUpstream, Reply};
use crate::Registry;

fn file_provider(id: &str, base_url: &str, secret: &str, models: Vec<&str>) -> ProviderConfig {
    ProviderConfig {
        id: id.into(),
        display_name: None,
        base_url: base_url.into(),
        format: Default::default(),
        auth: AuthConfig {
            scheme: Default::default(),
            secret: secret.into(),
            ..Default::default()
        },
        api_keys: vec![],
        key_strategy: None,
        headers: vec![],
        models: models
            .into_iter()
            .map(|m| ModelConfig {
                id: m.into(),
                name: None,
                context: None,
                max_output: None,
            })
            .collect(),
        passthrough: false,
    }
}

fn empty_registry() -> Registry {
    Registry::from_config(&AppConfig::minimal())
}

#[tokio::test]
async fn parses_feed_registers_public_provider_and_builds_daily_free() {
    let feed = serde_json::json!({
        "updated_at": "2026-11-18T00:00:00Z",
        "providers": [
            {"id":"catpub","base_url":"https://api.catpub.example/v1/chat/completions",
             "models":[{"id":"m1","context":8000}]}
        ],
        // ghost/nope does not resolve and must be filtered out of the chain.
        "daily_free_chain": ["catpub/m1", "ghost/nope"]
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let registry = empty_registry();
    let mut tracked = HashSet::new();

    let n = sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect("sync ok");
    assert_eq!(n, 1);

    let p = registry.provider("catpub").expect("registered");
    assert!(p.passthrough);
    assert_eq!(p.transport.format, WireFormat::OpenAi);
    assert!(matches!(p.auth, Auth::None));
    assert_eq!(p.models.len(), 1);
    assert_eq!(p.models[0].id, "m1");
    assert_eq!(p.models[0].context, Some(8000));

    let combo = registry.combo("daily-free").expect("combo created");
    assert_eq!(combo.models, vec!["catpub/m1".to_string()]);
    assert_eq!(registry.get_alias("free").as_deref(), Some("daily-free"));
}

#[tokio::test]
async fn existing_file_provider_keeps_endpoint_and_credentials() {
    let mut cfg = AppConfig::minimal();
    cfg.providers = vec![file_provider(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "env:OPENROUTER_API_KEY",
        vec!["existing-model"],
    )];
    let registry = Registry::from_config(&cfg);

    // The feed tries to redirect openrouter to a different host and advertises
    // a new model. The configured endpoint and key must survive untouched.
    let feed = serde_json::json!({
        "providers": [
            {"id":"openrouter","base_url":"https://attacker.example/v1/chat/completions",
             "models":[{"id":"new-model"}]}
        ],
        "daily_free_chain": []
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let mut tracked = HashSet::new();

    sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect("sync ok");

    let p = registry.provider("openrouter").expect("still present");
    assert_eq!(
        p.transport.base_url, "https://openrouter.ai/api/v1/chat/completions",
        "catalog must not redirect a configured provider"
    );
    match &p.auth {
        Auth::Bearer { key_pool } => {
            assert_eq!(
                key_pool.keys,
                vec![SecretRef::Env("OPENROUTER_API_KEY".to_string())],
                "configured credential reference must be preserved"
            );
        }
        other => panic!("expected bearer env auth, got {other:?}"),
    }
    let ids: Vec<&str> = p.models.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"existing-model"));
    assert!(ids.contains(&"new-model"));
}

#[tokio::test]
async fn catalog_merge_does_not_mask_hot_reloaded_config() {
    let mut cfg = AppConfig::minimal();
    cfg.providers = vec![file_provider(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "env:KEY_OLD",
        vec!["base-model"],
    )];
    let registry = Registry::from_config(&cfg);

    let feed = serde_json::json!({
        "providers": [
            {"id":"openrouter","base_url":"https://attacker.example/v1/chat/completions",
             "models":[{"id":"cat-model"}]}
        ],
        "daily_free_chain": []
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let client = reqwest::Client::new();
    let mut tracked = HashSet::new();

    sync_remote_catalog(&client, &registry, &mock.url(), &mut tracked)
        .await
        .expect("sync ok");
    let p = registry.provider("openrouter").expect("present");
    assert!(p.models.iter().any(|m| m.id == "cat-model"));
    assert_eq!(
        p.transport.base_url,
        "https://openrouter.ai/api/v1/chat/completions"
    );

    // A repeated sync must be idempotent (overlay stores the full advertised
    // list, not a delta that would erase prior merges).
    sync_remote_catalog(&client, &registry, &mock.url(), &mut tracked)
        .await
        .expect("second sync ok");
    assert!(registry
        .provider("openrouter")
        .unwrap()
        .models
        .iter()
        .any(|m| m.id == "cat-model"));

    // Hot reload rotates the key and changes the endpoint. The catalog merge
    // must NOT mask the fresh file config.
    let mut cfg2 = AppConfig::minimal();
    cfg2.providers = vec![file_provider(
        "openrouter",
        "https://new.openrouter.ai/api/v1/chat/completions",
        "env:KEY_NEW",
        vec!["base-model"],
    )];
    registry.reload(&cfg2);
    let p2 = registry
        .provider("openrouter")
        .expect("present after reload");
    assert_eq!(
        p2.transport.base_url, "https://new.openrouter.ai/api/v1/chat/completions",
        "hot reload must not be masked by a catalog merge"
    );
    match &p2.auth {
        Auth::Bearer { key_pool } => {
            assert_eq!(key_pool.keys, vec![SecretRef::Env("KEY_NEW".to_string())]);
        }
        other => panic!("expected rotated bearer env auth, got {other:?}"),
    }
    // The catalog model overlay still applies on top of the fresh config.
    assert!(p2.models.iter().any(|m| m.id == "cat-model"));
}

#[tokio::test]
async fn ssrf_rejects_private_and_non_http_base_urls() {
    let feed = serde_json::json!({
        "providers": [
            {"id":"loopback","base_url":"http://127.0.0.1:11434/v1/chat/completions","models":[]},
            {"id":"rfc1918","base_url":"http://10.0.0.5/v1/chat/completions","models":[]},
            {"id":"metadata","base_url":"http://169.254.169.254/latest/","models":[]},
            {"id":"v6loop","base_url":"http://[::1]/v1/chat/completions","models":[]},
            {"id":"localhost","base_url":"http://localhost:8080/v1","models":[]},
            {"id":"ftp","base_url":"ftp://example.com/v1","models":[]}
        ],
        "daily_free_chain": []
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let registry = empty_registry();
    let mut tracked = HashSet::new();

    let n = sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect("sync ok");
    assert_eq!(n, 0, "no private/non-http provider may be registered");
    for id in [
        "loopback",
        "rfc1918",
        "metadata",
        "v6loop",
        "localhost",
        "ftp",
    ] {
        assert!(registry.provider(id).is_none(), "{id} must be rejected");
    }
}

#[tokio::test]
async fn stale_providers_and_empty_chain_are_removed() {
    let first = serde_json::json!({
        "providers": [
            {"id":"catpub","base_url":"https://api.catpub.example/v1/chat/completions",
             "models":[{"id":"m1"}]}
        ],
        "daily_free_chain": ["catpub/m1"]
    })
    .to_string();
    let second = serde_json::json!({
        "providers": [],
        "daily_free_chain": []
    })
    .to_string();
    let mock =
        MockUpstream::sequence(vec![Reply::Json(200, first), Reply::Json(200, second)]).await;
    let registry = empty_registry();
    let client = reqwest::Client::new();
    let mut tracked = HashSet::new();

    sync_remote_catalog(&client, &registry, &mock.url(), &mut tracked)
        .await
        .expect("first sync");
    assert!(registry.provider("catpub").is_some());
    assert!(registry.combo("daily-free").is_some());

    sync_remote_catalog(&client, &registry, &mock.url(), &mut tracked)
        .await
        .expect("second sync");
    assert!(
        registry.provider("catpub").is_none(),
        "a provider absent from the latest feed must be removed"
    );
    assert!(
        registry.combo("daily-free").is_none(),
        "an empty daily_free_chain must remove the combo"
    );
    assert!(tracked.is_empty());
}

#[tokio::test]
async fn free_alias_is_not_overwritten_when_already_set() {
    let registry = empty_registry();
    registry.set_alias("free", "my-own-combo");
    let feed = serde_json::json!({
        "providers": [
            {"id":"catpub","base_url":"https://api.catpub.example/v1/chat/completions",
             "models":[{"id":"m1"}]}
        ],
        "daily_free_chain": ["catpub/m1"]
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let mut tracked = HashSet::new();

    sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect("sync ok");
    assert_eq!(
        registry.get_alias("free").as_deref(),
        Some("my-own-combo"),
        "an existing free alias must be respected"
    );
}

#[tokio::test]
async fn oversized_body_is_rejected_without_touching_registry() {
    let big = "x".repeat(crate::catalog::MAX_CATALOG_BYTES + 1);
    let mock = MockUpstream::json(200, "OK", &big).await;
    let registry = empty_registry();
    let mut tracked = HashSet::new();

    let err = sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect_err("oversized body must fail");
    assert!(err.contains("cap"), "unexpected error: {err}");
    assert!(registry.provider_ids().is_empty());
}

#[tokio::test]
async fn invalid_json_is_non_fatal() {
    let mock = MockUpstream::json(200, "OK", "{not valid json").await;
    let registry = empty_registry();
    let mut tracked = HashSet::new();

    let err = sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect_err("invalid json must fail");
    assert!(err.contains("parse"), "unexpected error: {err}");
    assert!(registry.provider_ids().is_empty());
}

#[tokio::test]
async fn parses_vibe_radar_native_catalog_shape() {
    let feed = serde_json::json!({
        "generated_at": 1795505461,
        "provider_models": [
            {"id":"inclusionai/ling-3.2-flash:free","provider":"openrouter",
             "model_id":"inclusionai/ling-3.2-flash:free","deal_type":"free",
             "context_length":131072,"supports_tools":true,
             "claim_url":"https://openrouter.ai/models/inclusionai/ling-3.2-flash:free"}
        ],
        "keyless_endpoints": [
            {"name":"Pollinations Text","base_url":"https://text.pollinations.ai/openai",
             "check_url":"https://text.pollinations.ai/models","models":["openai","mistral"],
             "notes":"OpenAI-compatible","added_at":1795110617,"last_verified":1795505461}
        ]
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let registry = empty_registry();
    let mut tracked = HashSet::new();

    let n = sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect("native Radar sync ok");
    assert_eq!(n, 1, "only the keyless endpoint can be registered");
    assert!(
        registry.provider("openrouter").is_none(),
        "provider_models must never create an anonymous OpenRouter provider"
    );
    let p = registry
        .provider("radar-pollinations-text")
        .expect("registered");
    assert_eq!(p.transport.base_url, "https://text.pollinations.ai/openai");
    assert!(matches!(p.auth, Auth::None));
    assert_eq!(
        registry.combo("daily-free").unwrap().models,
        vec![
            "radar-pollinations-text/openai",
            "radar-pollinations-text/mistral"
        ]
    );
}

#[tokio::test]
async fn radar_models_overlay_openrouter_and_prioritize_tools() {
    let mut cfg = AppConfig::minimal();
    cfg.providers = vec![file_provider(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "env:OPENROUTER_API_KEY",
        vec!["base-model"],
    )];
    let registry = Registry::from_config(&cfg);
    let feed = serde_json::json!({
        "generated_at": 1795505461,
        "provider_models": [
            {"id":"vendor/plain:free","context_length":64000,"supports_tools":false},
            {"id":"vendor/tool-a:free","context_length":128000,"supports_tools":true},
            {"id":"vendor/unknown:free","context_length":4_294_967_296u64},
            {"id":"vendor/tool-b:free","supports_tools":true},
            {"id":"vendor/plain:free","supports_tools":false}
        ],
        "keyless_endpoints": []
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let mut tracked = HashSet::new();

    sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect("sync ok");

    let p = registry.provider("openrouter").expect("still present");
    assert_eq!(
        p.transport.base_url, "https://openrouter.ai/api/v1/chat/completions",
        "catalog must not redirect configured OpenRouter transport"
    );
    match &p.auth {
        Auth::Bearer { key_pool } => assert_eq!(
            key_pool.keys,
            vec![SecretRef::Env("OPENROUTER_API_KEY".to_string())]
        ),
        other => panic!("expected bearer env auth, got {other:?}"),
    }
    assert!(p.models.iter().any(|m| m.id == "base-model"));
    assert_eq!(
        p.models
            .iter()
            .find(|m| m.id == "vendor/tool-a:free")
            .and_then(|m| m.context),
        Some(128000)
    );
    assert_eq!(
        p.models
            .iter()
            .find(|m| m.id == "vendor/unknown:free")
            .and_then(|m| m.context),
        None,
        "u64 context lengths above u32::MAX must be ignored"
    );
    assert_eq!(
        registry.combo("daily-free").unwrap().models,
        vec![
            "openrouter/vendor/tool-a:free",
            "openrouter/vendor/tool-b:free",
            "openrouter/vendor/plain:free",
            "openrouter/vendor/unknown:free",
        ]
    );
    assert!(
        tracked.is_empty(),
        "file-provider overlays are not runtime providers"
    );

    let mut reloaded = AppConfig::minimal();
    reloaded.providers = vec![file_provider(
        "openrouter",
        "https://new.openrouter.ai/api/v1/chat/completions",
        "env:OPENROUTER_KEY_ROTATED",
        vec!["reloaded-base-model"],
    )];
    registry.reload(&reloaded);
    let reloaded_provider = registry
        .provider("openrouter")
        .expect("present after reload");
    assert_eq!(
        reloaded_provider.transport.base_url,
        "https://new.openrouter.ai/api/v1/chat/completions"
    );
    match &reloaded_provider.auth {
        Auth::Bearer { key_pool } => assert_eq!(
            key_pool.keys,
            vec![SecretRef::Env("OPENROUTER_KEY_ROTATED".to_string())]
        ),
        other => panic!("expected rotated bearer env auth, got {other:?}"),
    }
    assert!(reloaded_provider
        .models
        .iter()
        .any(|m| m.id == "reloaded-base-model"));
    assert!(reloaded_provider
        .models
        .iter()
        .any(|m| m.id == "vendor/tool-a:free"));
}

#[tokio::test]
async fn radar_keyless_endpoints_normalize_urls_and_reject_ssrf() {
    let feed = serde_json::json!({
        "generated_at": 1795505461,
        "provider_models": [],
        "keyless_endpoints": [
            {"name":"Direct API","base_url":"https://direct.example/openai/","models":["m1"]},
            {"name":"Versioned API","base_url":"https://versioned.example/v1/","models":["m2"]},
            {"name":"Loopback","base_url":"http://127.0.0.1:11434/v1","models":["bad"]},
            {"name":"Private","base_url":"http://10.0.0.5/v1","models":["bad"]},
            {"name":"Link Local","base_url":"http://169.254.169.254/latest","models":["bad"]},
            {"name":"IPv6 Loopback","base_url":"http://[::1]/v1","models":["bad"]},
            {"name":"Localhost","base_url":"http://localhost:8080/v1","models":["bad"]},
            {"name":"FTP","base_url":"ftp://example.com/v1","models":["bad"]},
            {"name":"!!!","base_url":"https://empty-slug.example/v1","models":["bad"]}
        ]
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let registry = empty_registry();
    let mut tracked = HashSet::new();

    let n = sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect("sync ok");
    assert_eq!(n, 2);

    let direct = registry.provider("radar-direct-api").expect("registered");
    assert_eq!(direct.transport.base_url, "https://direct.example/openai");
    assert_eq!(direct.transport.format, WireFormat::OpenAi);
    assert!(direct.passthrough);
    assert!(matches!(direct.auth, Auth::None));
    assert_eq!(direct.models[0].id, "m1");

    let versioned = registry
        .provider("radar-versioned-api")
        .expect("registered");
    assert_eq!(
        versioned.transport.base_url,
        "https://versioned.example/v1/chat/completions"
    );
    for id in [
        "radar-loopback",
        "radar-private",
        "radar-link-local",
        "radar-ipv6-loopback",
        "radar-localhost",
        "radar-ftp",
    ] {
        assert!(registry.provider(id).is_none(), "{id} must be rejected");
    }
    assert_eq!(tracked.len(), 2);
}

#[tokio::test]
async fn radar_keyless_endpoint_does_not_shadow_file_provider() {
    let mut cfg = AppConfig::minimal();
    cfg.providers = vec![file_provider(
        "radar-configured",
        "https://configured.example/v1/chat/completions",
        "env:CONFIGURED_KEY",
        vec!["configured-model"],
    )];
    let registry = Registry::from_config(&cfg);
    let feed = serde_json::json!({
        "generated_at": 1795505461,
        "keyless_endpoints": [
            {"name":"Configured","base_url":"https://attacker.example/v1","models":["evil"]}
        ]
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let mut tracked = HashSet::new();

    let n = sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect("sync ok");
    assert_eq!(n, 0);
    let p = registry
        .provider("radar-configured")
        .expect("file provider remains");
    assert_eq!(
        p.transport.base_url,
        "https://configured.example/v1/chat/completions"
    );
    assert!(p.models.iter().any(|m| m.id == "configured-model"));
    assert!(!p.models.iter().any(|m| m.id == "evil"));
}

#[tokio::test]
async fn stale_radar_models_and_keyless_providers_are_removed() {
    let mut cfg = AppConfig::minimal();
    cfg.providers = vec![file_provider(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "env:OPENROUTER_API_KEY",
        vec!["base-model"],
    )];
    let registry = Registry::from_config(&cfg);
    let first = serde_json::json!({
        "generated_at": 1795505461,
        "provider_models": [{"id":"vendor/gone:free","supports_tools":true}],
        "keyless_endpoints": [
            {"name":"Gone API","base_url":"https://gone.example/v1","models":["gone"]}
        ]
    })
    .to_string();
    let second = serde_json::json!({
        "generated_at": 1795505462,
        "provider_models": [],
        "keyless_endpoints": []
    })
    .to_string();
    let mock =
        MockUpstream::sequence(vec![Reply::Json(200, first), Reply::Json(200, second)]).await;
    let client = reqwest::Client::new();
    let mut tracked = HashSet::new();

    sync_remote_catalog(&client, &registry, &mock.url(), &mut tracked)
        .await
        .expect("first sync");
    assert!(registry.provider("radar-gone-api").is_some());
    assert!(registry
        .provider("openrouter")
        .unwrap()
        .models
        .iter()
        .any(|m| m.id == "vendor/gone:free"));

    sync_remote_catalog(&client, &registry, &mock.url(), &mut tracked)
        .await
        .expect("empty sync");
    assert!(registry.provider("radar-gone-api").is_none());
    let p = registry
        .provider("openrouter")
        .expect("file provider remains");
    assert!(p.models.iter().any(|m| m.id == "base-model"));
    assert!(!p.models.iter().any(|m| m.id == "vendor/gone:free"));
    assert!(registry.combo("daily-free").is_none());
    assert!(tracked.is_empty());
    assert!(registry.catalog_overlay_ids().is_empty());
}

#[tokio::test]
async fn mixed_catalog_sources_dedupe_candidates_in_order() {
    let mut cfg = AppConfig::minimal();
    cfg.providers = vec![file_provider(
        "openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        "env:OPENROUTER_API_KEY",
        vec!["base-model"],
    )];
    let registry = Registry::from_config(&cfg);
    let feed = serde_json::json!({
        "providers": [
            {"id":"legacy","base_url":"https://legacy.example/v1/chat/completions",
             "models":[{"id":"legacy-model"}]}
        ],
        "daily_free_chain": ["legacy/legacy-model", "openrouter/vendor/plain:free"],
        "provider_models": [
            {"id":"vendor/plain:free","supports_tools":false},
            {"id":"vendor/tool:free","supports_tools":true}
        ],
        "keyless_endpoints": [
            {"name":"Free API","base_url":"https://free.example/v1","models":["free-model"]}
        ]
    })
    .to_string();
    let mock = MockUpstream::json(200, "OK", &feed).await;
    let mut tracked = HashSet::new();

    sync_remote_catalog(
        &reqwest::Client::new(),
        &registry,
        &mock.url(),
        &mut tracked,
    )
    .await
    .expect("mixed sync ok");
    assert_eq!(
        registry.combo("daily-free").unwrap().models,
        vec![
            "legacy/legacy-model",
            "openrouter/vendor/plain:free",
            "openrouter/vendor/tool:free",
            "radar-free-api/free-model",
        ]
    );
}
