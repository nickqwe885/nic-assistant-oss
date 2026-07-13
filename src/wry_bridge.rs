//! Rust-side XSS gate for the wry/WebView2 bridge.
//!
//! The frontend is vanilla JS with no React VDOM, so HTML-escaping of any text
//! that originates outside the app (OCR captured off the screen, or the model's
//! answer) must happen on the Rust side of the boundary. Every string handed to
//! `webview.evaluate_script` is meant to pass through [`sanitize_to_html`] first,
//! so a `<`, `"` or `'` in captured text can never become live markup.
//!
//! Today model/OCR text reaches the UI over the HTTP+token API as JSON, not via
//! `evaluate_script`, so the active escape point is the `renderMarkdown`/`esc`
//! pass in ui.html. This gate is the canonical escape for when that text moves
//! onto the native IPC path; kept here, defined and tested, ready to wire in.

/// Escapes the five HTML-significant characters so `text` is safe to interpolate
/// into HTML before `webview.evaluate_script`.
///
/// `&` is replaced first so the entities introduced for the other characters are
/// not themselves re-escaped.
pub fn sanitize_to_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

#[cfg(test)]
mod tests {
    use super::sanitize_to_html;

    #[test]
    fn escapes_all_five_significant_chars() {
        assert_eq!(
            sanitize_to_html(r#"<a href="x" o='y'>&"#),
            "&lt;a href=&quot;x&quot; o=&#x27;y&#x27;&gt;&amp;"
        );
    }

    #[test]
    fn ampersand_is_escaped_first_no_double_encoding() {
        // The `&` in the entities we emit for `<` etc. must stay single-encoded.
        assert_eq!(sanitize_to_html("<"), "&lt;");
        assert_eq!(sanitize_to_html("&amp;"), "&amp;amp;");
    }

    #[test]
    fn neutralizes_the_link_regex_breakout() {
        // The ui.html:846 vector: a `"` inside a markdown link URL used to break
        // out of href="...". Once escaped it can no longer close the attribute.
        let malicious = r#"https://x"onmouseover=alert(1)"#;
        assert!(!sanitize_to_html(malicious).contains('"'));
    }

    #[test]
    fn leaves_plain_text_untouched() {
        assert_eq!(sanitize_to_html("hello world 123"), "hello world 123");
    }
}
