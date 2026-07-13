/// Detects the operating system's configured UI language.
/// Returns a full English language name suitable for LLM prompts.
pub fn detect_language() -> String {
    detect_impl()
}

/// Detects the user's city from the OS timezone setting.
/// Returns None if the timezone cannot be mapped to a known city.
pub fn detect_city() -> Option<String> {
    detect_city_impl()
}

#[cfg(windows)]
fn detect_city_impl() -> Option<String> {
    use winreg::enums::*;
    use winreg::RegKey;
    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    let key = hklm
        .open_subkey("SYSTEM\\CurrentControlSet\\Control\\TimeZoneInformation")
        .ok()?;
    let tz: String = key.get_value("TimeZoneKeyName").ok()?;
    timezone_to_city(&tz).map(str::to_string)
}

#[cfg(not(windows))]
fn detect_city_impl() -> Option<String> {
    None
}

fn timezone_to_city(tz: &str) -> Option<&'static str> {
    match tz {
        // CIS
        "Russian Standard Time"          => Some("Moscow"),
        "Russian Time Zone 1"            => Some("Samara"),
        "Ekaterinburg Standard Time"     => Some("Yekaterinburg"),
        "N. Central Asia Standard Time"  => Some("Omsk"),
        "Central Asia Standard Time"     => Some("Almaty"),
        "Kazakhstan Standard Time"       => Some("Astana"),
        "North Asia Standard Time"       => Some("Krasnoyarsk"),
        "North Asia East Standard Time"  => Some("Irkutsk"),
        "Transbaikal Standard Time"      => Some("Chita"),
        "Yakutsk Standard Time"          => Some("Yakutsk"),
        "Vladivostok Standard Time"      => Some("Vladivostok"),
        "Russia Time Zone 10"            => Some("Magadan"),
        "Russia Time Zone 11"            => Some("Petropavlovsk-Kamchatsky"),
        "FLE Standard Time"              => Some("Kyiv"),
        "Ukraine Standard Time"          => Some("Kyiv"),
        "Belarus Standard Time"          => Some("Minsk"),
        "Azerbaijan Standard Time"       => Some("Baku"),
        "Georgian Standard Time"         => Some("Tbilisi"),
        "Armenian Standard Time"         => Some("Yerevan"),
        "Uzbekistan Standard Time"       => Some("Tashkent"),
        "Kyrgyzstan Standard Time"       => Some("Bishkek"),
        "Tajikistan Standard Time"       => Some("Dushanbe"),
        "Turkmenistan Standard Time"     => Some("Ashgabat"),
        // Europe
        "GMT Standard Time"              => Some("London"),
        "W. Europe Standard Time"        => Some("Berlin"),
        "Central European Standard Time" => Some("Warsaw"),
        "Central Europe Standard Time"   => Some("Prague"),
        "Romance Standard Time"          => Some("Paris"),
        "Turkey Standard Time"           => Some("Istanbul"),
        "Israel Standard Time"           => Some("Tel Aviv"),
        // Asia
        "Arabian Standard Time"          => Some("Dubai"),
        "Arab Standard Time"             => Some("Riyadh"),
        "India Standard Time"            => Some("Delhi"),
        "China Standard Time"            => Some("Beijing"),
        "Tokyo Standard Time"            => Some("Tokyo"),
        "Korea Standard Time"            => Some("Seoul"),
        "Singapore Standard Time"        => Some("Singapore"),
        // Americas
        "Eastern Standard Time"          => Some("New York"),
        "Central Standard Time"          => Some("Chicago"),
        "Mountain Standard Time"         => Some("Denver"),
        "Pacific Standard Time"          => Some("Los Angeles"),
        "E. South America Standard Time" => Some("Sao Paulo"),
        // Africa
        "E. Africa Standard Time"        => Some("Nairobi"),
        "South Africa Standard Time"     => Some("Johannesburg"),
        _                                => None,
    }
}

#[cfg(windows)]
fn detect_impl() -> String {
    use winreg::enums::*;
    use winreg::RegKey;
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    if let Ok(key) = hkcu.open_subkey("Control Panel\\International") {
        if let Ok(locale) = key.get_value::<String, _>("LocaleName") {
            return locale_to_name(&locale).to_string();
        }
    }
    "English".to_string()
}

#[cfg(not(windows))]
fn detect_impl() -> String {
    // LANG=ru_RU.UTF-8 or LANGUAGE=de_DE → use first 2 chars
    let raw = std::env::var("LANGUAGE")
        .or_else(|_| std::env::var("LANG"))
        .unwrap_or_default();
    locale_to_name(raw.as_str()).to_string()
}

fn locale_to_name(locale: &str) -> &'static str {
    match locale.get(..2).unwrap_or("en") {
        "ru" => "Russian",
        "kk" => "Kazakh",
        "de" => "German",
        "fr" => "French",
        "es" => "Spanish",
        "zh" => "Chinese",
        "ja" => "Japanese",
        "uk" => "Ukrainian",
        "pl" => "Polish",
        "tr" => "Turkish",
        "it" => "Italian",
        "pt" => "Portuguese",
        "ar" => "Arabic",
        "ko" => "Korean",
        "nl" => "Dutch",
        "sv" => "Swedish",
        "da" => "Danish",
        "fi" => "Finnish",
        "cs" => "Czech",
        "sk" => "Slovak",
        "hu" => "Hungarian",
        "ro" => "Romanian",
        "bg" => "Bulgarian",
        "he" => "Hebrew",
        "vi" => "Vietnamese",
        "th" => "Thai",
        "id" => "Indonesian",
        _    => "English",
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_prefixes_map_correctly() {
        assert_eq!(locale_to_name("ru-RU"), "Russian");
        assert_eq!(locale_to_name("en-US"), "English");
        assert_eq!(locale_to_name("de-DE"), "German");
        assert_eq!(locale_to_name("fr-FR"), "French");
        assert_eq!(locale_to_name("zh-CN"), "Chinese");
        assert_eq!(locale_to_name("uk-UA"), "Ukrainian");
    }

    #[test]
    fn unknown_locale_falls_back_to_english() {
        assert_eq!(locale_to_name("xx-XX"), "English");
        assert_eq!(locale_to_name(""), "English");
    }

    #[test]
    fn detect_language_returns_non_empty() {
        assert!(!detect_language().is_empty());
    }

    // ── timezone_to_city ──────────────────────────────────────────────────────

    #[test]
    fn timezone_kazakhstan_maps_to_astana() {
        assert_eq!(timezone_to_city("Kazakhstan Standard Time"), Some("Astana"));
    }

    #[test]
    fn timezone_russian_standard_maps_to_moscow() {
        assert_eq!(timezone_to_city("Russian Standard Time"), Some("Moscow"));
    }

    #[test]
    fn timezone_tokyo_maps() {
        assert_eq!(timezone_to_city("Tokyo Standard Time"), Some("Tokyo"));
    }

    #[test]
    fn timezone_eastern_maps_to_new_york() {
        assert_eq!(timezone_to_city("Eastern Standard Time"), Some("New York"));
    }

    #[test]
    fn timezone_unknown_returns_none() {
        assert_eq!(timezone_to_city("Atlantis Standard Time"), None);
    }

    #[test]
    fn timezone_empty_returns_none() {
        assert_eq!(timezone_to_city(""), None);
    }

    // ── detect_city ───────────────────────────────────────────────────────────

    #[test]
    fn detect_city_returns_option_string() {
        // Just verify it doesn't panic and returns a valid Option<String>
        let city = detect_city();
        if let Some(ref c) = city {
            assert!(!c.is_empty(), "detected city should not be empty string");
        }
    }
}
