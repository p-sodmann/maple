//! Shared EXIF datetime parser.
//!
//! Parses the `"YYYY:MM:DD HH:MM:SS"` format used in EXIF DateTimeOriginal
//! fields.  Used by both `maple-db` (for Unix timestamps) and `maple-import`
//! (for folder-name formatting).

/// Parsed components of an EXIF datetime string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExifDateTime {
    pub year: u32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl ExifDateTime {
    /// Parse an EXIF datetime string `"YYYY:MM:DD HH:MM:SS"`.
    ///
    /// Returns `None` on any parse failure rather than panicking.
    pub fn parse(s: &str) -> Option<Self> {
        if s.len() < 19 {
            return None;
        }
        Some(Self {
            year: s[0..4].parse().ok()?,
            month: s[5..7].parse().ok()?,
            day: s[8..10].parse().ok()?,
            hour: s[11..13].parse().ok()?,
            minute: s[14..16].parse().ok()?,
            second: s[17..19].parse().ok()?,
        })
    }

    /// Convert to a Unix timestamp (seconds since 1970-01-01 UTC).
    ///
    /// No timezone adjustment — treats the EXIF time as UTC.
    /// Algorithm: <https://howardhinnant.github.io/date_algorithms.html#days_from_civil>
    pub fn to_unix_timestamp(&self) -> i64 {
        let year = self.year as i64;
        let month = self.month as i64;
        let day = self.day as i64;

        let y = if month <= 2 { year - 1 } else { year };
        let m = if month <= 2 { month + 9 } else { month - 3 };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let doy = (153 * m + 2) / 5 + day - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let days = era * 146097 + doe - 719468;

        days * 86400 + self.hour as i64 * 3600 + self.minute as i64 * 60 + self.second as i64
    }

    /// Format the datetime using strftime-style tokens (`%Y`, `%m`, `%d`).
    pub fn format(&self, fmt: &str) -> String {
        fmt.replace("%Y", &format!("{:04}", self.year))
            .replace("%m", &format!("{:02}", self.month))
            .replace("%d", &format!("{:02}", self.day))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid() {
        let dt = ExifDateTime::parse("2024:03:15 14:30:45").unwrap();
        assert_eq!(dt.year, 2024);
        assert_eq!(dt.month, 3);
        assert_eq!(dt.day, 15);
        assert_eq!(dt.hour, 14);
        assert_eq!(dt.minute, 30);
        assert_eq!(dt.second, 45);
    }

    #[test]
    fn parse_too_short() {
        assert!(ExifDateTime::parse("2024:03:15").is_none());
    }

    #[test]
    fn unix_timestamp_epoch() {
        let dt = ExifDateTime::parse("1970:01:01 00:00:00").unwrap();
        assert_eq!(dt.to_unix_timestamp(), 0);
    }

    #[test]
    fn unix_timestamp_known_date() {
        // 2024-01-01 00:00:00 UTC = 1704067200
        let dt = ExifDateTime::parse("2024:01:01 00:00:00").unwrap();
        assert_eq!(dt.to_unix_timestamp(), 1704067200);
    }

    #[test]
    fn format_strftime() {
        let dt = ExifDateTime::parse("2024:03:15 14:30:45").unwrap();
        assert_eq!(dt.format("%Y/%m"), "2024/03");
        assert_eq!(dt.format("%Y-%m-%d"), "2024-03-15");
    }
}
