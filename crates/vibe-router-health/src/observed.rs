//! Provider-reported rate limits, observed from upstream response headers.
//!
//! A pure, allocation-light parser: it knows the header dialects spoken by
//! OpenAI / OpenRouter / Groq (generic `x-ratelimit-*`), Anthropic
//! (`anthropic-ratelimit-*`, RFC 3339 reset timestamps), and the standard
//! `Retry-After` hint, and normalizes them into one [`ObservedLimit`].
//!
//! Safety: every derived cooldown is clamped to
//! [`MAX_OBSERVED_COOLDOWN_MS`] (10 minutes), so a malformed or hostile
//! upstream header can never permanently brick a key or a provider. A
//! response whose headers carry nothing recognizable parses to `None` and
//! has zero behavioral impact.

use serde::{Deserialize, Serialize};

/// Hard ceiling for any header-derived cooldown: 10 minutes.
pub const MAX_OBSERVED_COOLDOWN_MS: u64 = 600_000;

/// Rate-limit state reported by an upstream provider on one response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedLimit {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit_requests: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remaining_requests: Option<u64>,
    /// Milliseconds until the provider says the window resets, derived from
    /// its reset header and clamped to [`MAX_OBSERVED_COOLDOWN_MS`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reset_after_ms: Option<u64>,
    /// Milliseconds from a `Retry-After` header, clamped the same way.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    /// When this observation was taken (Unix epoch milliseconds).
    pub observed_at_ms: u64,
}

impl ObservedLimit {
    /// The cooldown the upstream is asking for, if it stated one:
    /// `Retry-After` wins over the window-reset hint.
    pub fn cooldown_hint_ms(&self) -> Option<u64> {
        self.retry_after_ms.or(self.reset_after_ms)
    }

    /// Parse an `ObservedLimit` from response headers. Header names are
    /// matched case-insensitively. Returns `None` when no recognized header
    /// parses cleanly.
    pub fn from_headers<'a>(
        headers: impl IntoIterator<Item = (&'a str, &'a str)>,
        now_ms: u64,
    ) -> Option<Self> {
        let mut limit_requests = None;
        let mut remaining_requests = None;
        let mut reset_raw: Option<&str> = None;
        let mut retry_raw: Option<&str> = None;

        // Dialect-specific slots first, generic fallbacks second: iterate
        // once, filling the most specific empty slot for each quantity.
        let mut limit_generic = None;
        let mut remaining_generic = None;
        let mut reset_generic = None;

        for (name, value) in headers {
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match name.to_ascii_lowercase().as_str() {
                "x-ratelimit-limit-requests" | "anthropic-ratelimit-requests-limit" => {
                    if limit_requests.is_none() {
                        limit_requests = value.parse::<u64>().ok();
                    }
                }
                "x-ratelimit-limit" => {
                    if limit_generic.is_none() {
                        limit_generic = value.parse::<u64>().ok();
                    }
                }
                "x-ratelimit-remaining-requests" | "anthropic-ratelimit-requests-remaining" => {
                    if remaining_requests.is_none() {
                        remaining_requests = value.parse::<u64>().ok();
                    }
                }
                "x-ratelimit-remaining" => {
                    if remaining_generic.is_none() {
                        remaining_generic = value.parse::<u64>().ok();
                    }
                }
                "x-ratelimit-reset-requests" | "anthropic-ratelimit-requests-reset" => {
                    reset_raw.get_or_insert(value);
                }
                "x-ratelimit-reset" => {
                    reset_generic.get_or_insert(value);
                }
                "retry-after" => {
                    retry_raw.get_or_insert(value);
                }
                _ => {}
            }
        }

        let limit_requests = limit_requests.or(limit_generic);
        let remaining_requests = remaining_requests.or(remaining_generic);
        let reset_after_ms = reset_raw
            .or(reset_generic)
            .and_then(|v| parse_reset_ms(v, now_ms))
            .map(|ms| ms.min(MAX_OBSERVED_COOLDOWN_MS));
        let retry_after_ms = retry_raw
            .and_then(|v| v.parse::<u64>().ok())
            .map(|secs| secs.saturating_mul(1000).min(MAX_OBSERVED_COOLDOWN_MS));

        if limit_requests.is_none()
            && remaining_requests.is_none()
            && reset_after_ms.is_none()
            && retry_after_ms.is_none()
        {
            return None;
        }
        Some(Self {
            limit_requests,
            remaining_requests,
            reset_after_ms,
            retry_after_ms,
            observed_at_ms: now_ms,
        })
    }
}

/// Parse a rate-limit *reset* value into milliseconds from `now_ms`.
/// Accepts compound durations (`"1m30s"`), plain seconds (`"30"`, `"1.5"`),
/// Unix epoch seconds / milliseconds, and RFC 3339 UTC timestamps.
fn parse_reset_ms(raw: &str, now_ms: u64) -> Option<u64> {
    let v = raw.trim();
    if v.is_empty() {
        return None;
    }
    if v.contains('T') {
        if let Some(epoch_ms) = parse_rfc3339_ms(v) {
            return Some(epoch_ms.saturating_sub(now_ms));
        }
    }
    if let Some(ms) = parse_duration_ms(v) {
        return Some(ms);
    }
    // Numeric: epoch timestamps are distinguished from plain seconds by
    // magnitude (epoch seconds >= 1e9, epoch ms >= 1e12).
    let n: f64 = v.parse().ok()?;
    if !n.is_finite() || n < 0.0 {
        return None;
    }
    if n >= 1_000_000_000_000.0 {
        Some((n as u64).saturating_sub(now_ms))
    } else if n >= 1_000_000_000.0 {
        Some((n as u64).saturating_mul(1000).saturating_sub(now_ms))
    } else {
        Some((n * 1000.0).round() as u64)
    }
}

/// Parse a compound duration string (`"250ms"`, `"1.5s"`, `"1m30s"`,
/// `"2h1m5s"`) into milliseconds. Returns `None` unless the whole string is
/// one or more `<number><unit>` components.
fn parse_duration_ms(v: &str) -> Option<u64> {
    let bytes = v.as_bytes();
    let mut i = 0;
    let mut total: f64 = 0.0;
    let mut components = 0u32;
    while i < bytes.len() {
        // Number part (integer or decimal).
        let start = i;
        let mut seen_digit = false;
        let mut seen_dot = false;
        while i < bytes.len() {
            match bytes[i] {
                b'0'..=b'9' => {
                    seen_digit = true;
                    i += 1;
                }
                b'.' if !seen_dot => {
                    seen_dot = true;
                    i += 1;
                }
                _ => break,
            }
        }
        if !seen_digit {
            return None;
        }
        let num: f64 = v[start..i].parse().ok()?;
        // Unit part: longest match wins ("ms" before "m").
        let (unit_ms, len) = if v[i..].starts_with("ms") {
            (1.0, 2)
        } else if v[i..].starts_with('s') {
            (1000.0, 1)
        } else if v[i..].starts_with('m') {
            (60_000.0, 1)
        } else if v[i..].starts_with('h') {
            (3_600_000.0, 1)
        } else {
            return None;
        };
        i += len;
        total += num * unit_ms;
        components += 1;
        // Tolerate spaces between components ("1m 30s").
        while i < bytes.len() && bytes[i] == b' ' {
            i += 1;
        }
    }
    if components == 0 {
        return None;
    }
    Some(total.round() as u64)
}

/// Parse an RFC 3339 / ISO-8601 timestamp (`2026-11-27T12:01:05Z`, optional
/// fractional seconds, `Z` or `±HH:MM` offset) into Unix epoch milliseconds.
/// Pure arithmetic (days-from-civil); no date crate.
fn parse_rfc3339_ms(v: &str) -> Option<u64> {
    let b = v.as_bytes();
    if b.len() < 20 {
        return None;
    }
    let num = |from: usize, len: usize| -> Option<i64> { v.get(from..from + len)?.parse().ok() };
    let year = num(0, 4)?;
    if b.get(4) != Some(&b'-') || b.get(7) != Some(&b'-') {
        return None;
    }
    let month = num(5, 2)?;
    let day = num(8, 2)?;
    if b.get(10) != Some(&b'T') && b.get(10) != Some(&b' ') {
        return None;
    }
    let hour = num(11, 2)?;
    if b.get(13) != Some(&b':') || b.get(16) != Some(&b':') {
        return None;
    }
    let minute = num(14, 2)?;
    let second = num(17, 2)?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let mut idx = 19;
    let mut frac_ms: i64 = 0;
    if b.get(idx) == Some(&b'.') {
        idx += 1;
        let start = idx;
        let mut scale = 100i64;
        while let Some(c) = b.get(idx) {
            if !c.is_ascii_digit() {
                break;
            }
            if scale >= 1 {
                frac_ms += i64::from(c - b'0') * scale;
                scale /= 10;
            }
            idx += 1;
        }
        if idx == start {
            return None;
        }
    }
    // Zone: Z or ±HH:MM.
    let offset_ms: i64 = match b.get(idx) {
        Some(b'Z') | Some(b'z') => {
            idx += 1;
            0
        }
        Some(sign @ (b'+' | b'-')) => {
            let oh = num(idx + 1, 2)?;
            if b.get(idx + 3) != Some(&b':') {
                return None;
            }
            let om = num(idx + 4, 2)?;
            idx += 6;
            let mag = (oh * 3600 + om * 60) * 1000;
            if *sign == b'+' {
                mag
            } else {
                -mag
            }
        }
        _ => return None,
    };
    if idx != b.len() {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let epoch_ms = (((days * 24 + hour) * 60 + minute) * 60 + second) * 1000 + frac_ms - offset_ms;
    if epoch_ms < 0 {
        return None;
    }
    Some(epoch_ms as u64)
}

/// Days since 1970-01-01 for a proleptic Gregorian date
/// (Howard Hinnant's days-from-civil algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-11-27T12:00:00Z in epoch milliseconds.
    const NOW: u64 = 1_795_780_800_000;

    fn parse(headers: Vec<(&str, &str)>) -> Option<ObservedLimit> {
        ObservedLimit::from_headers(headers, NOW)
    }

    #[test]
    fn openai_duration_strings() {
        let o = parse(vec![
            ("x-ratelimit-limit-requests", "500"),
            ("x-ratelimit-remaining-requests", "499"),
            ("x-ratelimit-reset-requests", "250ms"),
        ])
        .unwrap();
        assert_eq!(o.limit_requests, Some(500));
        assert_eq!(o.remaining_requests, Some(499));
        assert_eq!(o.reset_after_ms, Some(250));
        assert_eq!(o.observed_at_ms, NOW);

        let o = parse(vec![("x-ratelimit-reset-requests", "1m30s")]).unwrap();
        assert_eq!(o.reset_after_ms, Some(90_000));

        let o = parse(vec![("x-ratelimit-reset-requests", "1.5s")]).unwrap();
        assert_eq!(o.reset_after_ms, Some(1500));
    }

    #[test]
    fn generic_numeric_seconds_and_fallbacks() {
        // Generic header names work as fallbacks for the -requests variants.
        let o = parse(vec![
            ("x-ratelimit-limit", "100"),
            ("x-ratelimit-remaining", "0"),
            ("x-ratelimit-reset", "30"),
        ])
        .unwrap();
        assert_eq!(o.limit_requests, Some(100));
        assert_eq!(o.remaining_requests, Some(0));
        assert_eq!(o.reset_after_ms, Some(30_000));

        // The -requests variant wins over the generic fallback.
        let o = parse(vec![
            ("x-ratelimit-limit", "100"),
            ("x-ratelimit-limit-requests", "500"),
        ])
        .unwrap();
        assert_eq!(o.limit_requests, Some(500));
    }

    #[test]
    fn epoch_timestamps_distinguished_from_seconds() {
        // Epoch seconds: NOW is 1795780800 s; +60 s.
        let o = parse(vec![("x-ratelimit-reset-requests", "1795780860")]).unwrap();
        assert_eq!(o.reset_after_ms, Some(60_000));
        // Epoch milliseconds: NOW + 45 s.
        let o = parse(vec![("x-ratelimit-reset-requests", "1795780845000")]).unwrap();
        assert_eq!(o.reset_after_ms, Some(45_000));
        // A past epoch clamps to zero, not a negative duration.
        let o = parse(vec![("x-ratelimit-reset-requests", "1795780700")]).unwrap();
        assert_eq!(o.reset_after_ms, Some(0));
    }

    #[test]
    fn anthropic_rfc3339_reset() {
        let o = parse(vec![
            ("anthropic-ratelimit-requests-limit", "1000"),
            ("anthropic-ratelimit-requests-remaining", "999"),
            ("anthropic-ratelimit-requests-reset", "2026-11-27T12:01:05Z"),
        ])
        .unwrap();
        assert_eq!(o.limit_requests, Some(1000));
        assert_eq!(o.remaining_requests, Some(999));
        assert_eq!(o.reset_after_ms, Some(65_000));

        // Fractional seconds.
        let o = parse(vec![(
            "anthropic-ratelimit-requests-reset",
            "2026-11-27T12:00:01.500Z",
        )])
        .unwrap();
        assert_eq!(o.reset_after_ms, Some(1500));

        // A +01:00 offset denotes the same instant as 12:01:05Z.
        let o = parse(vec![(
            "anthropic-ratelimit-requests-reset",
            "2026-11-27T13:01:05+01:00",
        )])
        .unwrap();
        assert_eq!(o.reset_after_ms, Some(65_000));
    }

    #[test]
    fn retry_after_seconds() {
        let o = parse(vec![("Retry-After", "45")]).unwrap();
        assert_eq!(o.retry_after_ms, Some(45_000));
        assert_eq!(o.cooldown_hint_ms(), Some(45_000));
        // HTTP-date form is not an integer: ignored cleanly.
        assert!(parse(vec![("Retry-After", "Wed, 21 Oct 2015 07:28:00 GMT")]).is_none());
    }

    #[test]
    fn cooldowns_clamped_to_ten_minutes() {
        let o = parse(vec![("x-ratelimit-reset-requests", "2h1m5s")]).unwrap();
        assert_eq!(o.reset_after_ms, Some(MAX_OBSERVED_COOLDOWN_MS));
        let o = parse(vec![("retry-after", "3600")]).unwrap();
        assert_eq!(o.retry_after_ms, Some(MAX_OBSERVED_COOLDOWN_MS));
        let o = parse(vec![("x-ratelimit-reset-requests", "99999999999")]).unwrap();
        assert_eq!(o.reset_after_ms, Some(MAX_OBSERVED_COOLDOWN_MS));
    }

    #[test]
    fn case_insensitive_names() {
        let o = parse(vec![
            ("X-RateLimit-Remaining-Requests", "7"),
            ("RETRY-AFTER", "3"),
        ])
        .unwrap();
        assert_eq!(o.remaining_requests, Some(7));
        assert_eq!(o.retry_after_ms, Some(3000));
    }

    #[test]
    fn malformed_and_empty_headers_yield_none() {
        assert!(parse(vec![]).is_none());
        assert!(parse(vec![("content-type", "application/json")]).is_none());
        assert!(parse(vec![("x-ratelimit-reset-requests", "soon")]).is_none());
        assert!(parse(vec![("x-ratelimit-remaining-requests", "-1")]).is_none());
        assert!(parse(vec![("x-ratelimit-remaining-requests", "")]).is_none());
        assert!(parse(vec![(
            "anthropic-ratelimit-requests-reset",
            "2026-13-40T99:99:99Z"
        )])
        .is_none());
    }

    #[test]
    fn serialization_omits_absent_fields() {
        let o = parse(vec![("x-ratelimit-remaining-requests", "0")]).unwrap();
        let v = serde_json::to_value(&o).unwrap();
        assert_eq!(v["remaining_requests"], 0);
        assert!(v.get("limit_requests").is_none());
        assert!(v.get("reset_after_ms").is_none());
        assert_eq!(v["observed_at_ms"], NOW);
    }
}
