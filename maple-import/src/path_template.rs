//! Folder/filename template rendering for `copy_images`.
//!
//! Templates use `{TOKEN}` placeholders resolved against a per-file
//! [`TemplateContext`]. Unknown tokens are left as literal text (so a typo
//! doesn't silently swallow part of the template). Folder templates are
//! `/`-separated; each segment is rendered and sanitized independently, and
//! empty segments (e.g. produced by an absent `{camera}`) are dropped.

use std::path::PathBuf;

use crate::ExifDateTime;

/// Per-file context used to resolve template tokens.
pub struct TemplateContext<'a> {
    /// Capture date/time — from EXIF `DateTimeOriginal`, or the file's
    /// mtime when no EXIF date is available. `None` only when neither is
    /// obtainable, in which case date/time tokens render as empty strings.
    pub datetime: Option<ExifDateTime>,
    /// Source filename stem (no extension).
    pub original_stem: &'a str,
    /// 1-based index of this file within the current copy batch.
    pub counter: usize,
    /// EXIF Make+Model, sanitized, when present.
    pub camera: Option<&'a str>,
}

impl TemplateContext<'_> {
    fn resolve_token(&self, token: &str) -> Option<String> {
        if let Some(dt) = &self.datetime {
            match token {
                "YYYY" => return Some(format!("{:04}", dt.year)),
                "YY" => return Some(format!("{:02}", dt.year % 100)),
                "MM" => return Some(format!("{:02}", dt.month)),
                "DD" => return Some(format!("{:02}", dt.day)),
                "hh" => return Some(format!("{:02}", dt.hour)),
                "mm" => return Some(format!("{:02}", dt.minute)),
                "ss" => return Some(format!("{:02}", dt.second)),
                _ => {}
            }
        } else if matches!(token, "YYYY" | "YY" | "MM" | "DD" | "hh" | "mm" | "ss") {
            return Some(String::new());
        }

        match token {
            "original" => Some(self.original_stem.to_owned()),
            "counter" => Some(format!("{:04}", self.counter)),
            "camera" => Some(self.camera.unwrap_or("").to_owned()),
            _ => None,
        }
    }
}

/// Render a `{TOKEN}`-based folder template into a relative path.
///
/// Splits on `/`, renders and sanitizes each segment independently, and
/// drops any segment that ends up empty (e.g. an unresolved `{camera}`).
/// An empty template renders to an empty (flat) path.
pub fn render_folder(template: &str, ctx: &TemplateContext) -> PathBuf {
    let mut result = PathBuf::new();
    for segment in template.split('/') {
        let rendered = sanitize_component(&render(segment, ctx));
        if !rendered.is_empty() {
            result.push(rendered);
        }
    }
    result
}

/// Render a `{TOKEN}`-based filename stem template.
///
/// The result never includes an extension — callers always append the
/// source file's original extension after this.
pub fn render_filename_stem(template: &str, ctx: &TemplateContext) -> String {
    sanitize_component(&render(template, ctx))
}

fn render(template: &str, ctx: &TemplateContext) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after_open = &rest[open + 1..];
        match after_open.find('}') {
            Some(close) => {
                let token = &after_open[..close];
                match ctx.resolve_token(token) {
                    Some(value) => out.push_str(&value),
                    None => out.push_str(&rest[open..open + 2 + close]),
                }
                rest = &after_open[close + 1..];
            }
            None => {
                // Unmatched '{' — keep the rest of the string verbatim.
                out.push_str(&rest[open..]);
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Strip path separators and characters invalid in Windows/macOS/Linux
/// filenames, and trim trailing dots/spaces (invalid on Windows).
fn sanitize_component(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    cleaned.trim_end_matches(['.', ' ']).trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(datetime: Option<ExifDateTime>) -> TemplateContext<'static> {
        TemplateContext {
            datetime,
            original_stem: "IMG_1234",
            counter: 7,
            camera: Some("Fujifilm X100V"),
        }
    }

    fn sample_dt() -> ExifDateTime {
        ExifDateTime::parse("2024:03:15 14:30:45").unwrap()
    }

    #[test]
    fn folder_date_tokens() {
        let c = ctx(Some(sample_dt()));
        assert_eq!(render_folder("{YYYY}/{MM}/{DD}", &c), PathBuf::from("2024/03/15"));
        assert_eq!(render_folder("{YY}", &c), PathBuf::from("24"));
    }

    #[test]
    fn folder_empty_template_is_flat() {
        let c = ctx(Some(sample_dt()));
        assert_eq!(render_folder("", &c), PathBuf::new());
    }

    #[test]
    fn folder_drops_empty_segments() {
        // No datetime -> date tokens resolve to "", so that segment is dropped.
        let c = ctx(None);
        assert_eq!(render_folder("{YYYY}/{camera}", &c), PathBuf::from("Fujifilm X100V"));
    }

    #[test]
    fn filename_presets() {
        let c = ctx(Some(sample_dt()));
        assert_eq!(render_filename_stem("{original}", &c), "IMG_1234");
        assert_eq!(render_filename_stem("{YYYY}{MM}{DD}_{original}", &c), "20240315_IMG_1234");
        assert_eq!(
            render_filename_stem("{YYYY}{MM}{DD}_{hh}{mm}{ss}_{original}", &c),
            "20240315_143045_IMG_1234"
        );
        assert_eq!(render_filename_stem("{YYYY}{MM}{DD}_{counter}", &c), "20240315_0007");
    }

    #[test]
    fn filename_no_datetime_leaves_date_tokens_empty() {
        let c = ctx(None);
        assert_eq!(render_filename_stem("{YYYY}_{original}", &c), "_IMG_1234");
    }

    #[test]
    fn unknown_tokens_pass_through_literally() {
        let c = ctx(Some(sample_dt()));
        assert_eq!(render_filename_stem("{nope}_{original}", &c), "{nope}_IMG_1234");
    }

    #[test]
    fn unmatched_brace_kept_verbatim() {
        let c = ctx(Some(sample_dt()));
        assert_eq!(render_filename_stem("{original}_{oops", &c), "IMG_1234_{oops");
    }

    #[test]
    fn sanitizes_invalid_filesystem_characters() {
        let mut c = ctx(Some(sample_dt()));
        c.camera = Some("Canon: EOS/R5");
        assert_eq!(render_filename_stem("{camera}", &c), "Canon_ EOS_R5");
    }

    #[test]
    fn sanitizes_trailing_dots_and_spaces() {
        let c = ctx(Some(sample_dt()));
        assert_eq!(render_filename_stem("{original}.", &c), "IMG_1234");
    }
}
