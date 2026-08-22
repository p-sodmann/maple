//! What the sidebar pill says, and the cell it reads.
//!
//! # Why a shared cell rather than a channel
//!
//! Import and restructure progress travel by `mpsc`, because every one of
//! those events matters — a dropped "copied 412 of 2481" leaves a gap in the
//! log. A status pill is the opposite: it wants the *current level* and
//! nothing else. Queueing state transitions for a 1-second poller to drain
//! would mean the pill briefly showing states that stopped being true
//! minutes ago, and a worker blocked on a full channel. One
//! `Arc<Mutex<SyncStatus>>` overwritten in place has neither problem.
//!
//! P5's worker writes this cell; until then it holds whatever role the
//! database says, which is `Off` on any installation that has not been set up.
//!
//! # Why amber and red are different
//!
//! "Never set up" and "was working, now broken" need different reactions from
//! the user, and a single warning colour would collapse them. Amber means
//! *waiting* — the other machine has not answered yet, which during pairing
//! is the normal state. Red means *stop and do something*: either the master
//! is gone ([`SyncState::Offline`]) or the credential was rejected
//! ([`SyncState::Unauthorized`]), and only the second of those will still be
//! broken tomorrow.
//!
//! # Determinism
//!
//! [`SyncStatus::display`] is a pure function of the status value, and
//! [`SyncStatus::tooltip`] takes `now_ms` as an argument. Both are testable
//! without an event loop, which is the whole reason the label and colour
//! choice lives here rather than in `.slint` markup.

use std::sync::{Arc, Mutex};

pub use maple_state::SyncRole;

/// Where the link is, independent of this device's role.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SyncState {
    /// Sync is switched off for this installation.
    #[default]
    Off,
    /// Configured, but no peer has answered yet. On a master this is the
    /// steady state whenever no servant is connected.
    Connecting,
    /// Linked and nothing to do.
    Idle,
    /// A sync pass is running.
    Running { done: u32, total: u32 },
    /// The peer could not be reached. Carries the pending retry delay so the
    /// pill can say when it will try again rather than looking hung.
    Offline { retry_secs: u64 },
    /// The peer rejected our credential. Terminal until the user re-pairs —
    /// see [`crate::backoff`] for why this never becomes a retry.
    Unauthorized,
}

/// The four pill colours, as an enum so this crate need not depend on Slint.
///
/// The RGB values are semantic rather than themed: a red that shifts with the
/// light/dark palette would stop reading as an alarm in one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusTone {
    Grey,
    Amber,
    Green,
    Red,
}

impl StatusTone {
    pub fn rgb(self) -> (u8, u8, u8) {
        match self {
            Self::Grey => (0x8A, 0x83, 0x7A),
            Self::Amber => (0xD9, 0x9A, 0x2B),
            Self::Green => (0x3F, 0x8F, 0x5D),
            Self::Red => (0xC0, 0x39, 0x2B),
        }
    }
}

/// Everything the pill needs to render one state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusDisplay {
    pub label: String,
    pub tone: StatusTone,
    /// Whether the dot animates. Reserved for the states where something is
    /// actively in flight, so a still dot genuinely means "nothing happening".
    pub pulsing: bool,
}

/// The shared status cell. Written by the sync worker (P5), read by the
/// 1-second poller on the UI thread.
pub type StatusCell = Arc<Mutex<SyncStatus>>;

/// The current state of sync on this installation.
#[derive(Debug, Clone, Default)]
pub struct SyncStatus {
    pub role: SyncRole,
    pub state: SyncState,
    /// Peers currently linked. Only meaningful for a master, which is passive
    /// and may have several.
    pub peers_online: u32,
    /// When the last pass completed, in Unix milliseconds.
    pub last_sync_ms: Option<i64>,
    /// Rows waiting to be sent.
    pub pending: u32,
    /// Why the last pass failed, if it did.
    ///
    /// Kept out of the pill's *label* on purpose: `Offline · retry 30s` has
    /// to stay short enough to read in a 228 px sidebar, and a protocol
    /// mismatch or a bind failure is a sentence, not a word. It surfaces on
    /// hover instead, which is where a user who wants the reason looks — and
    /// it means a seventh state was not needed to carry one string.
    pub last_error: Option<String>,
}

impl SyncStatus {
    /// A freshly-configured status for `role`, before any worker has run.
    ///
    /// `Off` means the pill reads grey; anything else starts amber, because
    /// "configured but nothing has answered yet" is exactly `Connecting`.
    pub fn for_role(role: SyncRole) -> Self {
        Self {
            role,
            state: match role {
                SyncRole::Off => SyncState::Off,
                _ => SyncState::Connecting,
            },
            ..Self::default()
        }
    }

    /// A shared cell holding [`SyncStatus::for_role`].
    pub fn cell(role: SyncRole) -> StatusCell {
        Arc::new(Mutex::new(Self::for_role(role)))
    }

    /// Label, colour and animation for the sidebar pill.
    pub fn display(&self) -> StatusDisplay {
        // Role Off wins over whatever the state field happens to hold: a
        // worker shutting down can leave a stale state behind, and the pill
        // must not keep claiming a link the user has switched off.
        if self.role == SyncRole::Off {
            return StatusDisplay {
                label: "Sync off".into(),
                tone: StatusTone::Grey,
                pulsing: false,
            };
        }

        match &self.state {
            SyncState::Off => StatusDisplay {
                label: "Sync off".into(),
                tone: StatusTone::Grey,
                pulsing: false,
            },
            SyncState::Unauthorized => StatusDisplay {
                label: "Re-pair required".into(),
                tone: StatusTone::Red,
                pulsing: false,
            },
            SyncState::Offline { retry_secs } => StatusDisplay {
                label: format!("Offline · retry {retry_secs}s"),
                tone: StatusTone::Red,
                pulsing: false,
            },
            SyncState::Running { done, total } => StatusDisplay {
                label: format!("Syncing {done}/{total}"),
                tone: StatusTone::Green,
                pulsing: true,
            },
            // A master is passive: it does not connect to anything, it waits
            // to be connected to. "Connecting…" would be a lie, and so would
            // a green "0 devices" — both readings collapse into the one
            // honest statement, that it is listening and nobody is there.
            SyncState::Connecting | SyncState::Idle if self.role == SyncRole::Master => {
                if self.peers_online == 0 {
                    StatusDisplay {
                        label: "Listening · no devices".into(),
                        tone: StatusTone::Amber,
                        pulsing: false,
                    }
                } else {
                    StatusDisplay {
                        label: format!("{} {}", self.peers_online, plural_devices(self.peers_online)),
                        tone: StatusTone::Green,
                        pulsing: false,
                    }
                }
            }
            SyncState::Connecting => StatusDisplay {
                label: "Connecting…".into(),
                tone: StatusTone::Amber,
                pulsing: true,
            },
            SyncState::Idle => StatusDisplay {
                label: "Synced".into(),
                tone: StatusTone::Green,
                pulsing: false,
            },
        }
    }

    /// Hover text: when the last pass finished and how much is waiting.
    ///
    /// Takes `now_ms` rather than reading the clock so the relative time is
    /// reproducible in a test.
    pub fn tooltip(&self, now_ms: i64) -> String {
        let mut parts = Vec::new();
        parts.push(match self.last_sync_ms {
            Some(then) => format!("Last sync {}", relative_time(now_ms - then)),
            None => "Never synced".to_owned(),
        });
        if self.pending > 0 {
            parts.push(format!("{} pending", self.pending));
        }
        if let Some(error) = self.last_error.as_deref() {
            parts.push(error.to_owned());
        }
        parts.join(" · ")
    }
}

fn plural_devices(n: u32) -> &'static str {
    if n == 1 {
        "device"
    } else {
        "devices"
    }
}

/// "just now" / "4 min ago" / "3 h ago" / "2 d ago".
///
/// A negative age means the peer's clock runs ahead of ours, which is common
/// enough on a LAN not to deserve a scary rendering — it reads as "just now".
///
/// Public because the settings card's "last seen" column wants exactly this
/// wording; two spellings of the same idea in two places is how a UI ends up
/// saying "4 min ago" in one row and "4 minutes ago" in the next.
pub fn relative_time(age_ms: i64) -> String {
    const MINUTE: i64 = 60_000;
    const HOUR: i64 = 60 * MINUTE;
    const DAY: i64 = 24 * HOUR;

    match age_ms {
        ms if ms < MINUTE => "just now".to_owned(),
        ms if ms < HOUR => format!("{} min ago", ms / MINUTE),
        ms if ms < DAY => format!("{} h ago", ms / HOUR),
        ms => format!("{} d ago", ms / DAY),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn servant(state: SyncState) -> SyncStatus {
        SyncStatus {
            role: SyncRole::Servant,
            state,
            ..SyncStatus::default()
        }
    }

    fn master(state: SyncState, peers_online: u32) -> SyncStatus {
        SyncStatus {
            role: SyncRole::Master,
            state,
            peers_online,
            ..SyncStatus::default()
        }
    }

    #[test]
    fn servant_states_map_to_their_labels_and_colours() {
        let cases = [
            (SyncState::Off, "Sync off", StatusTone::Grey, false),
            (SyncState::Connecting, "Connecting…", StatusTone::Amber, true),
            (SyncState::Idle, "Synced", StatusTone::Green, false),
            (
                SyncState::Running {
                    done: 14,
                    total: 120,
                },
                "Syncing 14/120",
                StatusTone::Green,
                true,
            ),
            (
                SyncState::Offline { retry_secs: 30 },
                "Offline · retry 30s",
                StatusTone::Red,
                false,
            ),
            (
                SyncState::Unauthorized,
                "Re-pair required",
                StatusTone::Red,
                false,
            ),
        ];
        for (state, label, tone, pulsing) in cases {
            let display = servant(state.clone()).display();
            assert_eq!(display.label, label, "label for {state:?}");
            assert_eq!(display.tone, tone, "tone for {state:?}");
            assert_eq!(display.pulsing, pulsing, "pulse for {state:?}");
        }
    }

    #[test]
    fn master_mode_reads_differently_from_servant_mode() {
        // The three readings §1.3 gives a master: grey off, amber listening,
        // green device count. None of them is "Connecting…" or "Synced".
        assert_eq!(
            master(SyncState::Off, 0).display(),
            StatusDisplay {
                label: "Sync off".into(),
                tone: StatusTone::Grey,
                pulsing: false
            }
        );
        assert_eq!(
            master(SyncState::Connecting, 0).display(),
            StatusDisplay {
                label: "Listening · no devices".into(),
                tone: StatusTone::Amber,
                pulsing: false
            }
        );
        assert_eq!(
            master(SyncState::Idle, 2).display(),
            StatusDisplay {
                label: "2 devices".into(),
                tone: StatusTone::Green,
                pulsing: false
            }
        );
    }

    #[test]
    fn a_master_with_no_peers_is_never_green() {
        // `Idle` with nobody connected is the trap: it is "idle" in the
        // literal sense, but reporting it green would tell the user their
        // laptop is synced when it has not been seen in a week.
        let display = master(SyncState::Idle, 0).display();
        assert_eq!(display.label, "Listening · no devices");
        assert_eq!(display.tone, StatusTone::Amber);
    }

    #[test]
    fn one_device_is_singular() {
        assert_eq!(master(SyncState::Idle, 1).display().label, "1 device");
    }

    #[test]
    fn a_master_still_reports_a_rejected_credential() {
        let display = master(SyncState::Unauthorized, 0).display();
        assert_eq!(display.label, "Re-pair required");
        assert_eq!(display.tone, StatusTone::Red);
    }

    #[test]
    fn role_off_overrides_a_stale_state() {
        let stale = SyncStatus {
            role: SyncRole::Off,
            state: SyncState::Idle,
            peers_online: 3,
            ..SyncStatus::default()
        };
        assert_eq!(stale.display().label, "Sync off");
        assert_eq!(stale.display().tone, StatusTone::Grey);
    }

    #[test]
    fn for_role_starts_amber_unless_switched_off() {
        assert_eq!(SyncStatus::for_role(SyncRole::Off).state, SyncState::Off);
        assert_eq!(
            SyncStatus::for_role(SyncRole::Servant).state,
            SyncState::Connecting
        );
        assert_eq!(
            SyncStatus::for_role(SyncRole::Master).display().tone,
            StatusTone::Amber
        );
    }

    #[test]
    fn every_tone_has_a_distinct_colour() {
        let tones = [
            StatusTone::Grey,
            StatusTone::Amber,
            StatusTone::Green,
            StatusTone::Red,
        ];
        for (i, a) in tones.iter().enumerate() {
            for b in &tones[i + 1..] {
                assert_ne!(a.rgb(), b.rgb(), "{a:?} and {b:?} share a colour");
            }
        }
    }

    #[test]
    fn tooltip_reports_age_and_pending_count() {
        let now = 1_700_000_000_000;
        let status = SyncStatus {
            role: SyncRole::Servant,
            state: SyncState::Idle,
            last_sync_ms: Some(now - 4 * 60_000),
            pending: 14,
            ..SyncStatus::default()
        };
        assert_eq!(status.tooltip(now), "Last sync 4 min ago · 14 pending");
    }

    #[test]
    fn tooltip_omits_a_zero_pending_count_and_names_a_never_synced_device() {
        let now = 1_700_000_000_000;
        assert_eq!(SyncStatus::for_role(SyncRole::Servant).tooltip(now), "Never synced");
    }

    #[test]
    fn a_peer_clock_running_ahead_reads_as_just_now() {
        let now = 1_700_000_000_000;
        let status = SyncStatus {
            last_sync_ms: Some(now + 30_000),
            ..SyncStatus::for_role(SyncRole::Servant)
        };
        assert_eq!(status.tooltip(now), "Last sync just now");
    }

    #[test]
    fn the_tooltip_carries_the_failure_reason_the_label_has_no_room_for() {
        let now = 1_700_000_000_000;
        let status = SyncStatus {
            role: SyncRole::Servant,
            state: SyncState::Offline { retry_secs: 30 },
            last_error: Some("peer speaks sync protocol 2 but this build speaks 1".into()),
            ..SyncStatus::default()
        };
        assert_eq!(status.display().label, "Offline · retry 30s", "label stays short");
        assert!(
            status.tooltip(now).contains("protocol 2"),
            "the reason belongs on hover: {}",
            status.tooltip(now)
        );
    }

    #[test]
    fn relative_time_scales_through_its_units() {
        assert_eq!(relative_time(30_000), "just now");
        assert_eq!(relative_time(90_000), "1 min ago");
        assert_eq!(relative_time(3 * 60 * 60_000), "3 h ago");
        assert_eq!(relative_time(2 * 24 * 60 * 60_000), "2 d ago");
    }
}
