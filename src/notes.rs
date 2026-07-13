//! Idea-writer: turns a messy brain-dump into a clean note saved in a sandboxed
//! "NIC Notes" folder. Deliberately minimal and safe — it can ONLY create or
//! append a `.md` file inside that one folder. No arbitrary paths, no path
//! traversal, no silent delete or overwrite of unrelated files.

use anyhow::{bail, Result};
use std::path::PathBuf;

/// The single folder NIC may write notes into: `%USERPROFILE%\Documents\NIC Notes`
/// (falls back to the app data dir). Nothing outside this folder is ever touched.
pub fn notes_dir() -> PathBuf {
    if let Ok(up) = std::env::var("USERPROFILE") {
        return PathBuf::from(up).join("Documents").join("NIC Notes");
    }
    crate::config::nic_data_dir().join("NIC Notes")
}

/// Reduces a title to a safe file stem: keeps letters/digits (incl. Cyrillic),
/// spaces, `-` and `_`; drops everything else (so no `/`, `\`, `..`, `:` etc.).
fn sanitize_stem(title: &str) -> String {
    let cleaned: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { ' ' })
        .collect();
    let collapsed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed: String = collapsed.chars().take(50).collect();
    let trimmed = trimmed.trim().to_string();
    if trimmed.is_empty() { "idea".to_string() } else { trimmed }
}

/// Saves a note: creates `NIC Notes/<title>.md`, or appends a dated section if a
/// note with that title already exists. Returns the file name. Never escapes the
/// sandbox folder and never deletes/overwrites existing content.
pub fn save_note(title: &str, body: &str) -> Result<String> {
    let dir = notes_dir();
    std::fs::create_dir_all(&dir)?;

    let stem  = sanitize_stem(title);
    let fname = format!("{stem}.md");
    // Belt-and-suspenders: the sanitized stem cannot contain separators, but
    // refuse outright if anything slipped through.
    if fname.contains('/') || fname.contains('\\') || fname.contains("..") {
        bail!("unsafe note filename");
    }
    let path = dir.join(&fname);

    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M");
    if path.exists() {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new().append(true).open(&path)?;
        write!(f, "\n\n---\n_{ts}_\n{body}\n")?;
    } else {
        std::fs::write(&path, format!("# {title}\n_{ts}_\n\n{body}\n"))?;
    }
    Ok(fname)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_separators_and_traversal() {
        assert_eq!(sanitize_stem("../../etc/passwd"), "etc passwd");
        assert_eq!(sanitize_stem("a/b\\c:d*e"), "a b c d e");
        assert_eq!(sanitize_stem("  идея про физику  "), "идея про физику");
    }

    #[test]
    fn sanitize_empty_becomes_idea() {
        assert_eq!(sanitize_stem(""), "idea");
        assert_eq!(sanitize_stem("///"), "idea");
    }

    #[test]
    fn sanitize_caps_length() {
        let long = "a".repeat(200);
        assert!(sanitize_stem(&long).chars().count() <= 50);
    }
}
