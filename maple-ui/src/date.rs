//! Minimal proleptic-Gregorian date math for Unix timestamps.
//!
//! No `chrono`/`time` dependency — Maple only ever needs UTC calendar dates
//! for display (EXIF "Taken" field, library date-grouping headers).

const WEEKDAYS: [&str; 7] =
    ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

/// Days since the Unix epoch (1970-01-01), floor-divided so timestamps
/// anywhere within a UTC calendar day map to the same value.
pub fn day_number(unix_ts: i64) -> i64 {
    unix_ts.div_euclid(86400)
}

/// Civil (Gregorian) year/month/day for a day count since the Unix epoch.
/// Howard Hinnant's `civil_from_days` algorithm.
pub fn days_to_ymd(days: u64) -> (u32, u32, u32) {
    let z = days + 719468;
    let era = z / 146097;
    let doe = z % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y as u32, mo as u32, d as u32)
}

/// A human-readable day header, e.g. `"Tuesday, 30.06.2026"`, for the
/// library's date-grouped view. Timestamps before the epoch clamp to day 0.
pub fn day_label(unix_ts: i64) -> String {
    let days = day_number(unix_ts).max(0) as u64;
    let (y, mo, d) = days_to_ymd(days);
    let weekday = WEEKDAYS[((days + 4) % 7) as usize];
    format!("{weekday}, {d:02}.{mo:02}.{y:04}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_thursday() {
        assert_eq!(day_label(0), "Thursday, 01.01.1970");
    }

    #[test]
    fn known_date() {
        // 2026-06-30 12:00:00 UTC
        assert_eq!(day_label(1782820800), "Tuesday, 30.06.2026");
    }

    #[test]
    fn same_day_different_times_share_a_day_number() {
        let start_of_day = 1782777600; // 2026-06-30 00:00:00 UTC
        let end_of_day = 1782863999; // 2026-06-30 23:59:59 UTC
        assert_eq!(day_number(start_of_day), day_number(end_of_day));
    }
}
