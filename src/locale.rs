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
        "Russian Standard Time"          => Some("Москва"),
        "Russian Time Zone 1"            => Some("Самара"),
        "Ekaterinburg Standard Time"     => Some("Екатеринбург"),
        "N. Central Asia Standard Time"  => Some("Омск"),
        "Central Asia Standard Time"     => Some("Алматы"),
        "Kazakhstan Standard Time"       => Some("Астана"),
        "North Asia Standard Time"       => Some("Красноярск"),
        "North Asia East Standard Time"  => Some("Иркутск"),
        "Transbaikal Standard Time"      => Some("Чита"),
        "Yakutsk Standard Time"          => Some("Якутск"),
        "Vladivostok Standard Time"      => Some("Владивосток"),
        "Russia Time Zone 10"            => Some("Магадан"),
        "Russia Time Zone 11"            => Some("Петропавловск-Камчатский"),
        "FLE Standard Time"              => Some("Киев"),
        "Ukraine Standard Time"          => Some("Киев"),
        "Belarus Standard Time"          => Some("Минск"),
        "Azerbaijan Standard Time"       => Some("Баку"),
        "Georgian Standard Time"         => Some("Тбилиси"),
        "Armenian Standard Time"         => Some("Ереван"),
        "Uzbekistan Standard Time"       => Some("Ташкент"),
        "Kyrgyzstan Standard Time"       => Some("Бишкек"),
        "Tajikistan Standard Time"       => Some("Душанбе"),
        "Turkmenistan Standard Time"     => Some("Ашхабад"),
        // Europe
        "GMT Standard Time"              => Some("Лондон"),
        "W. Europe Standard Time"        => Some("Берлин"),
        "Central European Standard Time" => Some("Варшава"),
        "Central Europe Standard Time"   => Some("Прага"),
        "Romance Standard Time"          => Some("Париж"),
        "Turkey Standard Time"           => Some("Стамбул"),
        "Israel Standard Time"           => Some("Тель-Авив"),
        // Asia
        "Arabian Standard Time"          => Some("Дубай"),
        "Arab Standard Time"             => Some("Эр-Рияд"),
        "India Standard Time"            => Some("Дели"),
        "China Standard Time"            => Some("Пекин"),
        "Tokyo Standard Time"            => Some("Токио"),
        "Korea Standard Time"            => Some("Сеул"),
        "Singapore Standard Time"        => Some("Сингапур"),
        // Americas
        "Eastern Standard Time"          => Some("Нью-Йорк"),
        "Central Standard Time"          => Some("Чикаго"),
        "Mountain Standard Time"         => Some("Денвер"),
        "Pacific Standard Time"          => Some("Лос-Анджелес"),
        "E. South America Standard Time" => Some("Сан-Паулу"),
        // Africa
        "E. Africa Standard Time"        => Some("Найроби"),
        "South Africa Standard Time"     => Some("Йоханнесбург"),
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
        assert_eq!(timezone_to_city("Kazakhstan Standard Time"), Some("Астана"));
    }

    #[test]
    fn timezone_russian_standard_maps_to_moscow() {
        assert_eq!(timezone_to_city("Russian Standard Time"), Some("Москва"));
    }

    #[test]
    fn timezone_tokyo_maps() {
        assert_eq!(timezone_to_city("Tokyo Standard Time"), Some("Токио"));
    }

    #[test]
    fn timezone_eastern_maps_to_new_york() {
        assert_eq!(timezone_to_city("Eastern Standard Time"), Some("Нью-Йорк"));
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
