//! Registry of music / video services NIC can open, plus which media kind each
//! serves. The Librarian scores these against recent screen activity to learn
//! the user's preferred service ("SoundCloud person" vs "Spotify person"), so a
//! bare "play X" opens X where the user actually lives — no LLM in the loop.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Audio,
    Video,
    Both,
}

pub struct Service {
    pub id:      &'static str,
    pub label:   &'static str,
    /// Lowercase needles matched against `app_name + window_title` of memory
    /// events (browser tab titles usually carry the site name).
    pub aliases: &'static [&'static str],
    pub kind:    Kind,
    /// Search URL with `{q}` replaced by the url-encoded query.
    pub search:  &'static str,
    pub home:    &'static str,
}

pub const SERVICES: &[Service] = &[
    Service { id: "soundcloud", label: "SoundCloud",
        aliases: &["soundcloud", "саундклауд"], kind: Kind::Audio,
        search: "https://soundcloud.com/search?q={q}", home: "https://soundcloud.com" },
    Service { id: "spotify", label: "Spotify",
        aliases: &["spotify", "спотифай", "спотифи"], kind: Kind::Audio,
        search: "https://open.spotify.com/search/{q}", home: "https://open.spotify.com" },
    Service { id: "yandex_music", label: "Yandex Music",
        aliases: &["music.yandex", "яндекс музык", "yandex music"], kind: Kind::Audio,
        search: "https://music.yandex.ru/search?text={q}", home: "https://music.yandex.ru" },
    Service { id: "youtube_music", label: "YouTube Music",
        aliases: &["music.youtube", "youtube music"], kind: Kind::Audio,
        search: "https://music.youtube.com/search?q={q}", home: "https://music.youtube.com" },
    Service { id: "youtube", label: "YouTube",
        aliases: &["youtube", "ютуб", "youtu.be"], kind: Kind::Both,
        search: "https://www.youtube.com/results?search_query={q}", home: "https://www.youtube.com" },
    Service { id: "twitch", label: "Twitch",
        aliases: &["twitch", "твич"], kind: Kind::Video,
        search: "https://www.twitch.tv/search?term={q}", home: "https://www.twitch.tv" },
    Service { id: "netflix", label: "Netflix",
        aliases: &["netflix", "нетфликс"], kind: Kind::Video,
        search: "https://www.netflix.com/search?q={q}", home: "https://www.netflix.com" },
    Service { id: "kinopoisk", label: "Kinopoisk",
        aliases: &["kinopoisk", "кинопоиск"], kind: Kind::Video,
        search: "https://www.kinopoisk.ru/index.php?kp_query={q}", home: "https://www.kinopoisk.ru" },
    Service { id: "ivi", label: "ivi",
        aliases: &["ivi.ru", "иви"], kind: Kind::Video,
        search: "https://www.ivi.ru/search/?q={q}", home: "https://www.ivi.ru" },
];

/// Does a service of `svc` kind satisfy a request for `want`? `Both` (YouTube)
/// satisfies either; an exact-kind service only its own.
pub fn kind_matches(svc: Kind, want: Kind) -> bool {
    svc == want || svc == Kind::Both || want == Kind::Both
}

pub fn by_id(id: &str) -> Option<&'static Service> {
    SERVICES.iter().find(|s| s.id == id)
}

impl Service {
    /// Builds the search URL for `query` (url-encoded).
    pub fn search_url(&self, query: &str) -> String {
        self.search.replace("{q}", &urlencoding::encode(query))
    }
}
