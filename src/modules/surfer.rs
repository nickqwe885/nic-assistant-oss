use anyhow::Result;
use reqwest;
use scraper::{Html, Selector};
use tracing::{debug, info, warn};

const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

/// One web search result with its real source URL (for clickable citations).
#[derive(Debug, Clone)]
pub struct WebResult {
    pub title:   String,
    pub url:     String,
    pub snippet: String,
}

/// DuckDuckGo links are redirect wrappers like
/// `//duckduckgo.com/l/?uddg=<urlencoded real url>&rut=…`. Pull out and decode
/// the real destination so citations point at the actual source, not DDG.
fn decode_ddg_href(href: &str) -> String {
    let h = href.trim();
    if let Some(idx) = h.find("uddg=") {
        let rest = &h[idx + 5..];
        let enc  = rest.split('&').next().unwrap_or("");
        if let Ok(decoded) = urlencoding::decode(enc) {
            return decoded.into_owned();
        }
    }
    if let Some(stripped) = h.strip_prefix("//") {
        return format!("https://{stripped}");
    }
    h.to_string()
}

/// Renders results as the plain numbered text fed to the model (URLs omitted —
/// they're appended separately as clickable citations).
pub fn results_to_snippets(results: &[WebResult]) -> String {
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!("{}. {}: {}\n", i + 1, r.title, r.snippet));
    }
    out
}

/// True if the query is explicitly about VK — only then do we keep vk.com results.
fn wants_vk(query: &str) -> bool {
    let q = query.to_lowercase();
    q.contains("vk") || q.contains("вк") || q.contains("вконтакте")
}

/// Drop a result that points at VK unless the user actually asked about VK.
fn skip_vk(wants: bool, href: &str, title: &str) -> bool {
    !wants && (href.contains("vk.com") || title.to_lowercase().contains("вконтакте"))
}

/// Parse DuckDuckGo's main HTML endpoint (`html.duckduckgo.com/html/`).
/// Results are grouped in `.result` containers. Returns up to 5 numbered lines.
fn parse_ddg_html(html: &str, query: &str) -> Vec<WebResult> {
    let document = Html::parse_document(html);
    let (Ok(result_sel), Ok(title_sel), Ok(snippet_sel)) = (
        Selector::parse(".result"),
        Selector::parse(".result__a"),
        Selector::parse(".result__snippet"),
    ) else { return Vec::new() };

    let wants = wants_vk(query);
    let mut out: Vec<WebResult> = Vec::new();
    for result in document.select(&result_sel) {
        let title   = result.select(&title_sel).next()
            .map(|t| t.text().collect::<String>()).unwrap_or_default().trim().to_string();
        let href    = result.select(&title_sel).next()
            .and_then(|el| el.value().attr("href")).unwrap_or("");
        let snippet = result.select(&snippet_sel).next()
            .map(|s| s.text().collect::<String>()).unwrap_or_default().trim().to_string();
        if title.is_empty() && snippet.is_empty() { continue; }
        if skip_vk(wants, &href.to_lowercase(), &title) { continue; }
        out.push(WebResult { title, url: decode_ddg_href(href), snippet });
        if out.len() >= 5 { break; }
    }
    out
}

/// Parse DuckDuckGo Lite (`lite.duckduckgo.com/lite/`) — a flat table where
/// titles (`a.result-link`) and snippets (`.result-snippet`) are separate cells.
/// Used as a fallback when the main endpoint is throttled. Up to 5 numbered lines.
fn parse_ddg_lite(html: &str, query: &str) -> Vec<WebResult> {
    let document = Html::parse_document(html);
    let (Ok(link_sel), Ok(snip_sel)) = (
        Selector::parse("a.result-link"),
        Selector::parse(".result-snippet"),
    ) else { return Vec::new() };

    let titles: Vec<(String, String)> = document.select(&link_sel).map(|el| {
        let title = el.text().collect::<String>().trim().to_string();
        let href  = el.value().attr("href").unwrap_or("").to_string();
        (title, href)
    }).collect();
    let snippets: Vec<String> = document.select(&snip_sel)
        .map(|el| el.text().collect::<String>().trim().to_string()).collect();

    let wants = wants_vk(query);
    let mut out: Vec<WebResult> = Vec::new();
    for (i, (title, href)) in titles.iter().enumerate() {
        let snippet = snippets.get(i).cloned().unwrap_or_default();
        if title.is_empty() && snippet.is_empty() { continue; }
        if skip_vk(wants, &href.to_lowercase(), title) { continue; }
        out.push(WebResult { title: title.clone(), url: decode_ddg_href(href), snippet });
        if out.len() >= 5 { break; }
    }
    out
}

/// Blocking POST to a DuckDuckGo endpoint via ureq. Returns body or None.
fn ddg_post_sync(url: &str, query: &str) -> Option<String> {
    match ureq::post(url)
        .set("User-Agent", UA)
        .set("Accept-Language", "ru-RU,ru;q=0.9,en;q=0.8")
        .set("Accept", "text/html,application/xhtml+xml")
        .send_form(&[("q", query)])
    {
        Ok(r) => match r.into_string() {
            Ok(b)  => Some(b),
            Err(e) => { warn!("[Surfer/sync] read error ({}): {}", url, e); None }
        },
        Err(e) => { warn!("[Surfer/sync] fetch error ({}): {}", url, e); None }
    }
}

/// Synchronous DuckDuckGo snippet fetch — called from blocking context (spawn_blocking).
/// Tries the main HTML endpoint first, then falls back to Lite. Returns up to 5
/// result snippets as a plain-text string, or empty string on failure.
pub fn fetch_snippets_sync(query: &str) -> String {
    results_to_snippets(&fetch_web_results_sync(query))
}

/// Like `fetch_snippets_sync` but returns structured results (title + real URL +
/// snippet) so callers can render clickable source citations.
pub fn fetch_web_results_sync(query: &str) -> Vec<WebResult> {
    // Primary: HTML endpoint (POST — GET is blocked with an anomaly page).
    if let Some(body) = ddg_post_sync("https://html.duckduckgo.com/html/", query) {
        let out = parse_ddg_html(&body, query);
        if !out.is_empty() { return out; }
    }
    // Fallback: Lite endpoint — simpler markup, far less likely to be throttled.
    if let Some(body) = ddg_post_sync("https://lite.duckduckgo.com/lite/", query) {
        let out = parse_ddg_lite(&body, query);
        if !out.is_empty() { return out; }
    }
    Vec::new()
}

/// Resolves a YouTube search to its FIRST video's watch URL by scraping the
/// results page (no API key). Returns `None` on any failure so the caller can
/// fall back to opening the plain search page. Blocking — call off the runtime.
/// Words that mean "their newest upload", not part of the thing being searched.
/// "qewbite last video" searched literally and returned an Avengers trailer —
/// the filler polluted the query. Stripping it and sorting YouTube by upload date
/// returns what the user actually asked for: that creator's latest video.
const LATEST_MARKERS: &[&str] = &[
    "last video", "latest video", "newest video", "new video", "recent video",
    "последнее видео", "новое видео", "свежее видео",
];

pub fn first_youtube_watch_url(query: &str) -> Option<String> {
    let ql = query.to_lowercase();
    let wants_latest = LATEST_MARKERS.iter().any(|m| ql.contains(m));

    let search_term = if wants_latest {
        let mut t = ql.clone();
        for m in LATEST_MARKERS {
            t = t.replace(m, " ");
        }
        // Drop leftover connective words so only the creator's name is searched.
        let cleaned: Vec<&str> = t
            .split_whitespace()
            .filter(|w| !matches!(*w, "his" | "her" | "their" | "the" | "a" | "of" | "s"))
            .collect();
        let c = cleaned.join(" ").trim().to_string();
        if c.is_empty() { query.to_string() } else { c }
    } else {
        query.to_string()
    };

    let url = format!(
        "https://www.youtube.com/results?search_query={}{}",
        urlencoding::encode(&search_term),
        // sp=CAI%3D is YouTube's "Sort by: Upload date" filter (base64 "CAI="),
        // so the first hit is that creator's newest upload.
        if wants_latest { "&sp=CAI%3D" } else { "" }
    );
    let body = ureq::get(&url)
        .set("User-Agent", UA)
        .set("Accept-Language", "en-US,en;q=0.9")
        .call().ok()?
        .into_string().ok()?;
    // The results HTML embeds ytInitialData; the first `"videoId":"…"` (11 chars)
    // is the top result.
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r#""videoId":"([A-Za-z0-9_-]{11})""#).unwrap());
    let id = re.captures(&body)?.get(1)?.as_str();
    Some(format!("https://www.youtube.com/watch?v={}", id))
}

pub struct Surfer {
    client: reqwest::Client,
}

impl Surfer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
                .default_headers({
                    let mut h = reqwest::header::HeaderMap::new();
                    h.insert("Accept-Language", "ru-RU,ru;q=0.9,en;q=0.8".parse().unwrap());
                    h.insert("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8".parse().unwrap());
                    h
                })
                .build()?,
        })
    }

    /// Returns true only for queries that clearly need live internet data.
    /// Default is false — model answers from its own knowledge.
    fn needs_internet(query: &str) -> bool {
        let q = query.to_lowercase();

        let negation_block = [
            "не ищи", "не искать", "без интернет", "не интернет",
            "локально", "оффлайн", "только локально",
            // English-only beta (§9[1])
            "don't search", "do not search", "no internet", "without internet",
            "offline", "locally", "local only",
        ];
        if negation_block.iter().any(|&t| q.contains(t)) {
            debug!("[Surfer] Decision: Blocked by negation");
            return false;
        }

        let local_only = [
            // Identity
            "кто ты", "как тебя зовут", "твоё имя", "твое имя",
            "что ты", "ты умеешь", "ты можешь", "расскажи о себе",
            "твои возможности", "что умеешь",
            // Local context
            "посчитай", "напомни", "файл", "время", "дата",
            "экран", "терминал", "я писал", "в консоли",
            "моя версия", "у меня", "в системе", "версия",
            // General knowledge — model knows these
            "что такое", "что значит", "объясни", "как работает",
            // Coding / generation tasks
            "напиши", "сгенерируй", "придумай", "составь", "помоги",
            "пример", "как написать", "как сделать",
            // Math
            "сколько будет", "вычисли", "реши",
            // English-only beta (§9[1]) — mirror the categories above.
            "who are you", "what are you", "your name", "what can you do",
            "tell me about yourself", "what do you do",
            "remind", "my screen", "my terminal", "on my system", "my version",
            // NOTE: deliberately NOT "what is" — in English that also phrases realtime
            // queries ("what is the weather/price/population"), which must reach the
            // web. Pure definitions ("what is recursion") fall through to local anyway.
            "what's the meaning", "explain", "how does",
            "how do i", "write ", "generate", "create a", "help me",
            "example of", "how to", "calculate", "solve", "compute",
        ];
        if local_only.iter().any(|&t| q.contains(t)) {
            debug!("[Surfer] Decision: Blocked by local keywords");
            return false;
        }

        // Only go to web for high-confidence real-time signals
        let force_online = [
            // News / time-sensitive
            "новости", "сегодня",
            // Finance
            "курс", "акции", "биткоин", "крипто", "валют",
            // Weather
            "погода", "прогноз",
            // People / companies
            "биография", "состояние",
            "openai", "tesla", "apple", "nvidia", "spacex",
            // Explicit search commands
            "поищи", "найди в интернет", "погугли", "в интернете",
            // World facts / statistics
            "население", "численность", "сколько людей", "сколько человек",
            "население земли", "население планеты",
            "сколько стоит", "стоимость", "цена",
            "рекорд", "мировой рекорд",
            "когда основан", "год основания",
            "сколько стран", "сколько городов",
            // Follow-up precision requests
            "точная цифра", "точное число", "точные данные",
            "уточни", "уточните", "а точно", "а именно",
            "конкретная цифра", "конкретно сколько",
            // Historical facts
            "сколько длилась", "сколько продолжалась", "сколько длилось", "сколько шла",
            "когда умер", "когда родился", "когда родилась", "когда умерла",
            "когда началась", "когда закончилась", "когда произошло", "когда был",
            "сколько лет назад", "в каком году",
            // People and characters
            "кто такой", "кто такая", "кто это", "кто был", "кто была",
            "персонаж", "из аниме", "из сериала", "из фильма", "из манги", "из игры",
            "кто написал", "кто создал", "кто придумал", "кто изобрёл",
            // Misc facts
            "что за",
            // English-only beta (§9[1]) — realtime / factual signals.
            "news", "today", "latest", "right now", "current",
            "weather", "forecast",
            "price", "cost", "how much", "stock", "bitcoin", "crypto",
            "currency", "exchange rate", "net worth",
            "population", "how many people", "record", "world record",
            "who is", "who was", "who wrote", "who created", "who invented", "who founded",
            "when did", "when was", "when born", "when died", "what year",
            "biography", "founded in", "year founded",
            "search", "google", "look up", "on the internet", "search the web",
        ];
        if force_online.iter().any(|&t| q.contains(t)) {
            debug!("[Surfer] Decision: Force online keyword");
            return true;
        }

        debug!("[Surfer] Decision: No web trigger — answering locally");
        false
    }

    pub async fn maybe_search_web(&self, query: &str, force_offline: bool) -> Option<String> {
        if force_offline {
            return None;
        }
        if !Self::needs_internet(query) {
            return None;
        }
        self.search_web(query).await
    }

    /// Search unconditionally — the caller has already decided the web is needed.
    ///
    /// `needs_internet` is a keyword allowlist ("who is", "price", "news"…), and a
    /// bare name trips none of them. Live, "tell me about donk" therefore never
    /// reached the internet at all: the model answered from its own imagination
    /// (a Donkey Kong biography, then an invented Team Liquid player) while the one
    /// source that actually knew the answer was never consulted. When the API knows
    /// the user is asking about a person, it says so — and we go and look.
    pub async fn search_web(&self, query: &str) -> Option<String> {
        info!("[Surfer] Web search…");
        match self.fetch_snippets(query).await {
            Ok(snippets) if !snippets.is_empty() => {
                info!("[Surfer] Web search done ({} chars).", snippets.len());
                Some(format!("[Surfer] Web results:\n{}", snippets))
            }
            Ok(_) => {
                info!("[Surfer] Web search: no results.");
                None
            }
            Err(e) => {
                warn!("[Surfer] Search error: {}", e);
                None
            }
        }
    }

    #[cfg(test)]
    pub fn needs_internet_pub(query: &str) -> bool { Self::needs_internet(query) }

    /// Async POST to a DuckDuckGo endpoint via reqwest. Returns body or None.
    async fn ddg_post(&self, url: &str, query: &str) -> Option<String> {
        match self.client.post(url).form(&[("q", query)]).send().await {
            Ok(resp) => match resp.text().await {
                Ok(body) => Some(body),
                Err(e)   => { warn!("[Surfer] read error ({}): {}", url, e); None }
            },
            Err(e) => { warn!("[Surfer] fetch error ({}): {}", url, e); None }
        }
    }

    async fn fetch_snippets(&self, query: &str) -> Result<String> {
        // DuckDuckGo's HTML endpoint blocks GET with an "anomaly" page; it only
        // serves results to a POST form. When it is throttled (0 results), we
        // fall back to the Lite endpoint, which has stable, simple markup.
        info!("[Surfer] HTTP запрос к DuckDuckGo (html)…");
        if let Some(body) = self.ddg_post("https://html.duckduckgo.com/html/", query).await {
            let out = parse_ddg_html(&body, query);
            if !out.is_empty() {
                info!("[Surfer] Найдено {} сниппетов (html)", out.len());
                return Ok(results_to_snippets(&out));
            }
        }
        info!("[Surfer] html endpoint пуст — пробую lite…");
        if let Some(body) = self.ddg_post("https://lite.duckduckgo.com/lite/", query).await {
            let out = parse_ddg_lite(&body, query);
            info!("[Surfer] Найдено {} сниппетов (lite)", out.len());
            return Ok(results_to_snippets(&out));
        }
        info!("[Surfer] Найдено 0 сниппетов");
        Ok(String::new())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── needs_internet: negation blocks ──────────────────────────────────────

    #[test]
    fn negation_ne_ishi_blocks() {
        assert!(!Surfer::needs_internet_pub("не ищи в интернете"));
    }

    #[test]
    fn negation_bez_internet_blocks() {
        assert!(!Surfer::needs_internet_pub("ответь без интернета"));
    }

    #[test]
    fn negation_lokalno_blocks() {
        assert!(!Surfer::needs_internet_pub("локально найди ответ"));
    }

    #[test]
    fn negation_offlajn_blocks() {
        assert!(!Surfer::needs_internet_pub("работай оффлайн"));
    }

    // ── needs_internet: English (English-only beta) ──────────────────────────

    #[test]
    fn en_realtime_triggers_web() {
        assert!(Surfer::needs_internet_pub("what is the weather today"));
        assert!(Surfer::needs_internet_pub("latest news"));
        assert!(Surfer::needs_internet_pub("bitcoin price"));
        assert!(Surfer::needs_internet_pub("population of Japan"));
        assert!(Surfer::needs_internet_pub("who is Elon Musk"));
        assert!(Surfer::needs_internet_pub("how much does an iPhone cost"));
    }

    #[test]
    fn en_definitions_and_local_stay_offline() {
        assert!(!Surfer::needs_internet_pub("explain how TCP works"));
        assert!(!Surfer::needs_internet_pub("who are you"));
        assert!(!Surfer::needs_internet_pub("calculate 2 + 2"));
        assert!(!Surfer::needs_internet_pub("write a python function"));
        assert!(!Surfer::needs_internet_pub("answer offline please"));
    }

    // ── needs_internet: local-only blocks ────────────────────────────────────

    #[test]
    fn local_kto_ty_blocks() {
        assert!(!Surfer::needs_internet_pub("кто ты такой"));
    }

    #[test]
    fn local_chto_takoe_blocks() {
        assert!(!Surfer::needs_internet_pub("что такое машинное обучение"));
    }

    #[test]
    fn local_napishi_blocks() {
        assert!(!Surfer::needs_internet_pub("напиши мне функцию на python"));
    }

    #[test]
    fn local_skolko_budet_blocks() {
        assert!(!Surfer::needs_internet_pub("сколько будет 2+2"));
    }

    #[test]
    fn local_objasnj_blocks() {
        assert!(!Surfer::needs_internet_pub("объясни как работает TCP"));
    }

    #[test]
    fn local_vremja_blocks() {
        assert!(!Surfer::needs_internet_pub("который сейчас время"));
    }

    #[test]
    fn local_versija_blocks() {
        assert!(!Surfer::needs_internet_pub("версия rust у меня"));
    }

    #[test]
    fn local_primer_blocks() {
        assert!(!Surfer::needs_internet_pub("покажи пример сортировки"));
    }

    // ── needs_internet: force online ─────────────────────────────────────────

    #[test]
    fn online_novosti_forces() {
        assert!(Surfer::needs_internet_pub("последние новости сегодня"));
    }

    #[test]
    fn online_kurs_forces() {
        assert!(Surfer::needs_internet_pub("курс доллара к рублю"));
    }

    #[test]
    fn online_pogoda_forces() {
        assert!(Surfer::needs_internet_pub("погода в москве завтра"));
    }

    #[test]
    fn online_bitkoin_forces() {
        assert!(Surfer::needs_internet_pub("цена биткоина сейчас"));
    }

    #[test]
    fn online_openai_forces() {
        assert!(Surfer::needs_internet_pub("openai новая модель"));
    }

    #[test]
    fn online_nvidia_forces() {
        assert!(Surfer::needs_internet_pub("акции nvidia упали"));
    }

    #[test]
    fn online_naselenie_forces() {
        assert!(Surfer::needs_internet_pub("население земли сколько человек"));
    }

    #[test]
    fn online_stoimost_forces() {
        assert!(Surfer::needs_internet_pub("сколько стоит iphone 16"));
    }

    #[test]
    fn online_rekord_forces() {
        assert!(Surfer::needs_internet_pub("мировой рекорд в плавании"));
    }

    #[test]
    fn online_kogda_rodilsja_forces() {
        assert!(Surfer::needs_internet_pub("когда родился пушкин"));
    }

    #[test]
    fn online_kto_takoi_forces() {
        assert!(Surfer::needs_internet_pub("кто такой илон маск"));
    }

    #[test]
    fn online_iz_anime_forces() {
        assert!(Surfer::needs_internet_pub("персонаж из аниме наруто"));
    }

    #[test]
    fn online_kogda_nachalas_forces() {
        assert!(Surfer::needs_internet_pub("когда началась вторая мировая война"));
    }

    #[test]
    fn online_tochnie_dannye_forces() {
        assert!(Surfer::needs_internet_pub("точные данные по инфляции"));
    }

    #[test]
    fn online_uточni_forces() {
        assert!(Surfer::needs_internet_pub("уточни данные о населении"));
    }

    #[test]
    fn online_kto_napisal_forces() {
        assert!(Surfer::needs_internet_pub("кто написал войну и мир"));
    }

    #[test]
    fn online_chto_za_forces() {
        assert!(Surfer::needs_internet_pub("что за компания palantir"));
    }

    // ── needs_internet: no trigger → false ───────────────────────────────────

    #[test]
    fn no_trigger_general_question_false() {
        assert!(!Surfer::needs_internet_pub("расскажи про гравитацию"));
    }

    #[test]
    fn no_trigger_empty_false() {
        assert!(!Surfer::needs_internet_pub(""));
    }

    #[test]
    fn no_trigger_greeting_false() {
        assert!(!Surfer::needs_internet_pub("привет как дела"));
    }

    // ── needs_internet: negation beats force_online ───────────────────────────

    #[test]
    fn negation_beats_force_online() {
        // "новости" would normally trigger online, but "без интернет" blocks first
        // Depends on evaluation order: negation checked first → false
        assert!(!Surfer::needs_internet_pub("новости без интернета"));
    }

    #[test]
    fn negation_beats_force_online_2() {
        assert!(!Surfer::needs_internet_pub("погода оффлайн"));
    }

    // ── needs_internet: local beats force_online ──────────────────────────────

    #[test]
    fn local_sегодня_interaction() {
        // "сегодня" is a force_online keyword, but "версия" is local_only
        // "сегодня версия" → local_only fires first → false
        assert!(!Surfer::needs_internet_pub("сегодня версия программы"));
    }

    // ── fetch_snippets_sync: result is a string (network not required to not panic) ─

    #[test]
    fn fetch_snippets_sync_does_not_panic_on_empty_query() {
        // We don't assert content (needs network), just that it doesn't panic
        let result = std::panic::catch_unwind(|| {
            fetch_snippets_sync("")
        });
        assert!(result.is_ok(), "fetch_snippets_sync panicked on empty query");
    }

    #[test]
    fn fetch_snippets_sync_does_not_panic_on_long_query() {
        let long_q = "а".repeat(500);
        let result = std::panic::catch_unwind(|| {
            fetch_snippets_sync(&long_q)
        });
        assert!(result.is_ok(), "fetch_snippets_sync panicked on long query");
    }

    #[test]
    fn fetch_snippets_sync_does_not_panic_special_chars() {
        let result = std::panic::catch_unwind(|| {
            fetch_snippets_sync("test & query <foo> \"bar\"")
        });
        assert!(result.is_ok(), "fetch_snippets_sync panicked on special chars");
    }

    #[test]
    fn fetch_snippets_sync_returns_string() {
        // Result must be String (may be empty if network unavailable)
        let result: String = fetch_snippets_sync("rust programming");
        // Just assert it's a valid string (not checking content)
        let _ = result.len();
    }

    // ── parse_ddg_html / parse_ddg_lite: parsing without network ──────────────

    #[test]
    fn parse_html_extracts_results() {
        let html = r#"
        <div class="result">
          <a class="result__a" href="https://example.com">Example Title</a>
          <div class="result__snippet">Example snippet text</div>
        </div>
        <div class="result">
          <a class="result__a" href="https://foo.com">Foo</a>
          <div class="result__snippet">Bar snippet</div>
        </div>"#;
        let out = parse_ddg_html(html, "test");
        let txt = results_to_snippets(&out);
        assert!(txt.contains("Example Title"));
        assert!(txt.contains("Example snippet text"));
        assert!(txt.contains("Foo"));
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].url, "https://example.com"); // real URL captured
    }

    #[test]
    fn parse_html_filters_vk_when_not_wanted() {
        let html = r#"
        <div class="result">
          <a class="result__a" href="https://vk.com/foo">VK Page</a>
          <div class="result__snippet">vk snippet</div>
        </div>
        <div class="result">
          <a class="result__a" href="https://example.com">Good</a>
          <div class="result__snippet">good snippet</div>
        </div>"#;
        let txt = results_to_snippets(&parse_ddg_html(html, "погода")); // does not ask for VK
        assert!(!txt.contains("VK Page"));
        assert!(txt.contains("Good"));
    }

    #[test]
    fn parse_html_keeps_vk_when_wanted() {
        let html = r#"
        <div class="result">
          <a class="result__a" href="https://vk.com/foo">VK Page</a>
          <div class="result__snippet">vk snippet</div>
        </div>"#;
        let txt = results_to_snippets(&parse_ddg_html(html, "найди вконтакте foo"));
        assert!(txt.contains("VK Page"));
    }

    #[test]
    fn parse_html_empty_on_no_results() {
        assert!(parse_ddg_html("<html><body>nothing</body></html>", "x").is_empty());
    }

    #[test]
    fn decode_ddg_redirect_extracts_real_url() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=abc";
        assert_eq!(decode_ddg_href(href), "https://example.com/a");
        // Protocol-relative non-redirect → https
        assert_eq!(decode_ddg_href("//foo.com/x"), "https://foo.com/x");
        // Plain URL passes through
        assert_eq!(decode_ddg_href("https://bar.com"), "https://bar.com");
    }

    #[test]
    fn parse_lite_extracts_results() {
        let html = r#"
        <table>
          <tr><td><a class='result-link' href='https://example.com'>Lite Title</a></td></tr>
          <tr><td class='result-snippet'>Lite snippet text</td></tr>
          <tr><td><a class='result-link' href='https://foo.com'>Foo Lite</a></td></tr>
          <tr><td class='result-snippet'>Foo snippet</td></tr>
        </table>"#;
        let out = parse_ddg_lite(html, "test");
        let txt = results_to_snippets(&out);
        assert!(txt.contains("Lite Title"));
        assert!(txt.contains("Lite snippet text"));
        assert!(txt.contains("Foo Lite"));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn parse_lite_empty_on_no_results() {
        assert!(parse_ddg_lite("<html><body>nothing</body></html>", "x").is_empty());
    }
}
