//! Remote model-catalog synchronization (the `Vibe_Radar` feed).
//!
//! Both the legacy `providers` + `daily_free_chain` document and Vibe Radar's
//! `generated_at` + `provider_models` + `keyless_endpoints` document are
//! accepted. [`sync_remote_catalog`] fetches either shape on a bounded budget
//! and reconciles it into the [`Registry`]:
//!
//!   * **Credentials are preserved.** A provider id that already exists keeps
//!     its configured `transport` and `auth` (e.g. `env:OPENROUTER_API_KEY`);
//!     advertised models refresh catalog-owned providers or overlay file-backed
//!     and externally authored runtime providers.
//!   * **New catalog-only providers are SSRF-guarded** — only public `http://`
//!     / `https://` hosts are accepted — and registered as anonymous,
//!     passthrough OpenAI providers.
//!   * **Stale entries are removed** when they disappear from the feed.
//!   * **`daily-free`** is upserted as a runtime combo from the resolvable
//!     legacy chain, OpenRouter free models, and keyless endpoints, with a
//!     `free` alias when unset.
//!
//! Config remains the source of truth for auth and endpoints; the catalog only
//! fills in model metadata and free-tier chains. Reads are snapshot reads, so a
//! swap is atomic from the request path's point of view.
//!
//! Depends on: vibe-router-core, vibe-router-config (via the registry).

use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use vibe_router_core::{Auth, Combo, ModelRegistry, ModelSpec, Transport, WireFormat};

use crate::pool::ProviderKeyPool;
use crate::{ConfigProvider, Registry};

/// Hard cap on the catalog response body (2 MB).
pub const MAX_CATALOG_BYTES: usize = 2 * 1024 * 1024;
/// Bounded wall-clock budget for fetching the catalog.
pub const FETCH_TIMEOUT: Duration = Duration::from_secs(15);
/// Name of the auto-generated free-tier combo.
pub const DAILY_FREE_COMBO: &str = "daily-free";
/// Alias pointing at [`DAILY_FREE_COMBO`], registered only when unset.
pub const FREE_ALIAS: &str = "free";

/// The `Vibe_Radar` catalog feed. Legacy and Radar-native fields coexist so a
/// mixed document remains backwards compatible.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RemoteCatalog {
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub generated_at: Option<serde_json::Value>,
    #[serde(default)]
    pub providers: Vec<CatalogProvider>,
    #[serde(default)]
    pub daily_free_chain: Vec<String>,
    #[serde(default)]
    pub provider_models: Vec<RadarProviderModel>,
    #[serde(default)]
    pub keyless_endpoints: Vec<RadarKeylessEndpoint>,
}

/// A provider advertised by the catalog.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CatalogProvider {
    pub id: String,
    #[serde(default)]
    pub display_name: Option<String>,
    pub base_url: String,
    #[serde(default)]
    pub models: Vec<CatalogModel>,
}

/// A model advertised by the catalog.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CatalogModel {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub context: Option<u32>,
    #[serde(default)]
    pub max_output: Option<u32>,
}

impl CatalogModel {
    fn to_spec(&self) -> ModelSpec {
        ModelSpec {
            id: self.id.clone(),
            name: self.name.clone(),
            context: self.context,
            max_output: self.max_output,
            capabilities: Vec::new(),
        }
    }
}

/// An OpenRouter `:free` model advertised by Vibe Radar.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RadarProviderModel {
    pub id: String,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub deal_type: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub supports_tools: Option<bool>,
    #[serde(default)]
    pub claim_url: Option<String>,
}

impl RadarProviderModel {
    fn to_spec(&self) -> ModelSpec {
        ModelSpec {
            id: self.id.clone(),
            name: None,
            context: self.context_length.and_then(|c| u32::try_from(c).ok()),
            max_output: None,
            capabilities: Vec::new(),
        }
    }
}

/// A public keyless OpenAI-compatible endpoint advertised by Vibe Radar.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RadarKeylessEndpoint {
    pub name: String,
    pub base_url: String,
    pub models: Vec<String>,
}

/// Fetch the catalog at `url` and reconcile it into `registry`. Returns the
/// number of providers registered or merged. `tracked_providers` carries the
/// catalog-registered ids from earlier passes so stale ones can be removed.
///
/// Non-fatal by contract: any fetch/parse failure returns `Err` and leaves the
/// registry untouched, so the serve loop keeps the previous snapshot.
pub async fn sync_remote_catalog(
    client: &reqwest::Client,
    registry: &Registry,
    url: &str,
    tracked_providers: &mut HashSet<String>,
) -> Result<usize, String> {
    let resp = client
        .get(url)
        .timeout(FETCH_TIMEOUT)
        .send()
        .await
        .map_err(|e| format!("catalog fetch failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("catalog returned HTTP {}", resp.status()));
    }
    if let Some(len) = resp.content_length() {
        if len as usize > MAX_CATALOG_BYTES {
            return Err(format!("catalog body exceeds {MAX_CATALOG_BYTES} byte cap"));
        }
    }
    let mut buf: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("catalog read failed: {e}"))?;
        if buf.len() + chunk.len() > MAX_CATALOG_BYTES {
            return Err(format!("catalog body exceeds {MAX_CATALOG_BYTES} byte cap"));
        }
        buf.extend_from_slice(&chunk);
    }
    let catalog: RemoteCatalog =
        serde_json::from_slice(&buf).map_err(|e| format!("catalog parse failed: {e}"))?;

    let previous_overlays: HashSet<String> = registry.catalog_overlay_ids().into_iter().collect();
    let mut current: HashSet<String> = HashSet::new();
    let mut current_runtime = HashSet::new();
    let mut current_overlays = HashSet::new();
    let mut overlays: HashMap<String, Vec<ModelSpec>> = HashMap::new();
    for cp in &catalog.providers {
        if let Some(existing) = registry.provider(&cp.id) {
            if (tracked_providers.contains(&cp.id) || current_runtime.contains(&cp.id))
                && !registry.is_file_provider(&cp.id)
            {
                let mut refreshed = (*existing).clone();
                refreshed.models = cp.models.iter().map(CatalogModel::to_spec).collect();
                registry.set_runtime_provider(refreshed);
                current_runtime.insert(cp.id.clone());
            } else {
                // Existing file or externally authored runtime provider: overlay
                // only the advertised models. Configured transport, auth, and key
                // pool stay authoritative across catalog syncs and hot reloads.
                overlays
                    .entry(cp.id.clone())
                    .or_default()
                    .extend(cp.models.iter().map(CatalogModel::to_spec));
                current_overlays.insert(cp.id.clone());
            }
            current.insert(cp.id.clone());
            continue;
        }

        if !catalog_url_is_safe(&cp.base_url) {
            tracing::warn!(
                provider = %cp.id,
                "catalog provider rejected: base_url is not a public http(s) host"
            );
            continue;
        }
        registry.set_runtime_provider(ConfigProvider {
            id: cp.id.clone(),
            display_name: cp.display_name.clone().unwrap_or_else(|| cp.id.clone()),
            transport: Transport {
                base_url: cp.base_url.clone(),
                format: WireFormat::OpenAi,
                headers: Vec::new(),
            },
            auth: Auth::None,
            models: cp.models.iter().map(CatalogModel::to_spec).collect(),
            passthrough: true,
            key_pool: Arc::new(ProviderKeyPool::empty()),
        });
        current.insert(cp.id.clone());
        current_runtime.insert(cp.id.clone());
    }

    let openrouter_is_persistent = registry.provider("openrouter").is_some()
        && (registry.is_file_provider("openrouter")
            || !tracked_providers.contains("openrouter")
            || current_runtime.contains("openrouter"));
    if !catalog.provider_models.is_empty() && openrouter_is_persistent {
        overlays
            .entry("openrouter".to_string())
            .or_default()
            .extend(
                catalog
                    .provider_models
                    .iter()
                    .map(RadarProviderModel::to_spec),
            );
        current.insert("openrouter".to_string());
        current_overlays.insert("openrouter".to_string());
    }
    for (id, models) in overlays {
        registry.set_catalog_models(&id, models);
    }

    let mut keyless_candidates = Vec::new();
    for endpoint in &catalog.keyless_endpoints {
        let Some(slug) = radar_provider_slug(&endpoint.name) else {
            continue;
        };
        let id = format!("radar-{slug}");
        if registry.is_file_provider(&id) {
            tracing::warn!(
                provider = %id,
                "catalog keyless endpoint rejected: provider id is configured in the file"
            );
            continue;
        }
        let base_url = normalize_radar_base_url(&endpoint.base_url);
        if !catalog_url_is_safe(&base_url) {
            tracing::warn!(
                provider = %id,
                "catalog keyless endpoint rejected: base_url is not a public http(s) host"
            );
            continue;
        }
        let models: Vec<ModelSpec> = endpoint
            .models
            .iter()
            .map(|model| ModelSpec {
                id: model.clone(),
                name: None,
                context: None,
                max_output: None,
                capabilities: Vec::new(),
            })
            .collect();
        registry.set_runtime_provider(ConfigProvider {
            id: id.clone(),
            display_name: endpoint.name.clone(),
            transport: Transport {
                base_url,
                format: WireFormat::OpenAi,
                headers: Vec::new(),
            },
            auth: Auth::None,
            models,
            passthrough: true,
            key_pool: Arc::new(ProviderKeyPool::empty()),
        });
        current.insert(id.clone());
        current_runtime.insert(id.clone());
        keyless_candidates.extend(endpoint.models.iter().map(|model| format!("{id}/{model}")));
    }

    for id in previous_overlays.difference(&current_overlays) {
        registry.remove_catalog_models(id);
    }
    for id in tracked_providers.difference(&current_runtime) {
        registry.remove_runtime_provider(id);
    }
    let registered = current.len();
    *tracked_providers = current_runtime;

    // Auto-combo from the resolvable free-tier chain. Radar tool-capable models
    // lead the OpenRouter group while preserving feed order inside each group.
    let mut candidates = catalog.daily_free_chain.clone();
    for tool_capable in [true, false] {
        candidates.extend(
            catalog
                .provider_models
                .iter()
                .filter(|entry| match tool_capable {
                    true => entry.supports_tools == Some(true),
                    false => entry.supports_tools != Some(true),
                })
                .map(|entry| format!("openrouter/{}", entry.id)),
        );
    }
    candidates.extend(keyless_candidates);
    let mut seen = HashSet::new();
    let chain: Vec<String> = candidates
        .into_iter()
        .filter(|entry| seen.insert(entry.clone()) && registry.resolve(entry).is_ok())
        .collect();
    if chain.is_empty() {
        registry.remove_runtime_combo(DAILY_FREE_COMBO);
    } else {
        registry.set_runtime_combo(Combo {
            name: DAILY_FREE_COMBO.to_string(),
            description: Some("Auto-generated from the remote catalog".to_string()),
            models: chain,
            strategy: Default::default(),
        });
        if registry.get_alias(FREE_ALIAS).is_none() {
            registry.set_alias(FREE_ALIAS, DAILY_FREE_COMBO);
        }
    }

    Ok(registered)
}

fn radar_provider_slug(name: &str) -> Option<String> {
    let mut slug = String::new();
    let mut separator = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            if separator && !slug.is_empty() {
                slug.push('-');
            }
            slug.push(ch.to_ascii_lowercase());
            separator = false;
        } else if !slug.is_empty() {
            separator = true;
        }
    }
    (!slug.is_empty()).then_some(slug)
}

fn normalize_radar_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        trimmed.to_string()
    }
}

/// SSRF policy for catalog-only providers: only `http`/`https` URLs whose host
/// is not loopback, link-local, or private. Mirrors the admin-boundary guard;
/// local nodes are the job of [`crate::discovery`], not the public catalog.
fn catalog_url_is_safe(raw: &str) -> bool {
    let Ok(u) = url::Url::parse(raw) else {
        return false;
    };
    if !matches!(u.scheme(), "http" | "https") {
        return false;
    }
    let Some(host) = u.host_str() else {
        return false;
    };
    !host_is_blocked(host)
}

fn host_is_blocked(host: &str) -> bool {
    let h = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase();
    if h == "localhost" || h.ends_with(".localhost") {
        return true;
    }
    if let Ok(ip) = h.parse::<IpAddr>() {
        return match ip {
            IpAddr::V4(v4) => {
                v4.is_loopback()
                    || v4.is_link_local()
                    || v4.is_private()
                    || v4.is_unspecified()
                    || v4.is_broadcast()
            }
            IpAddr::V6(v6) => {
                let seg = v6.segments();
                v6.is_loopback()
                    || v6.is_unspecified()
                    || (seg[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
                    || (seg[0] & 0xfe00) == 0xfc00 // fc00::/7 unique-local
            }
        };
    }
    false
}
