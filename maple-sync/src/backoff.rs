//! Reconnect pacing for the servant worker.
//!
//! A servant that loses its master retries on a fixed escalating schedule —
//! 1, 2, 5, 15, 30, then 60 seconds forever — reset the moment a pass
//! succeeds. Short first steps heal the common case (the master was
//! rebooting, the Wi-Fi blipped) without waiting a minute; the cap stops a
//! laptop that has been off the network all week from drifting into
//! hour-long gaps.
//!
//! # The exception that matters
//!
//! An **authentication failure is not a network problem**. A rejected
//! credential — the peer was unpaired on the other side, the trust file was
//! restored from an old backup — will still be rejected in sixty seconds and
//! in six hours. Retrying it forever spends battery to keep showing the user
//! `Offline`, which is the wrong thing to tell them: the link is not down,
//! it is *revoked*, and only re-pairing fixes it.
//!
//! So [`Backoff::on_failure`] answers [`Retry::Never`] for
//! [`FailureKind::Auth`] and latches: every later call keeps answering
//! `Never` until [`Backoff::reset`] runs, which is what pairing does. The
//! status pill reads the same latch as `Re-pair required` rather than
//! `Offline · retry 30s`.
//!
//! Nothing here samples the clock. The worker asks for a delay and sleeps it
//! on its own stop-channel timeout, so the schedule is a pure function of how
//! many failures have happened.

use std::time::Duration;

/// The retry schedule, in seconds. The last entry is the cap and repeats.
pub const SCHEDULE_SECS: &[u64] = &[1, 2, 5, 15, 30, 60];

/// Why a sync pass failed. The distinction drives whether we retry at all,
/// so it is deliberately not a single "it broke" error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// The master could not be reached, or answered with something transient.
    /// Retryable — this is what backoff exists for.
    Unreachable,
    /// The master rejected our credential. Not retryable; needs a human.
    Auth,
}

/// What the worker should do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retry {
    /// Sleep this long, then try again.
    After(Duration),
    /// Do not try again. Only re-pairing clears this.
    Never,
}

impl Retry {
    /// The delay in whole seconds, or `None` for [`Retry::Never`]. The status
    /// pill's `retry 30s` reads this.
    pub fn secs(self) -> Option<u64> {
        match self {
            Self::After(d) => Some(d.as_secs()),
            Self::Never => None,
        }
    }
}

/// Escalating retry state for one peer.
#[derive(Debug, Clone, Default)]
pub struct Backoff {
    /// Index into [`SCHEDULE_SECS`] of the delay the *next* failure yields.
    step: usize,
    /// Set by an auth failure; latched until [`Backoff::reset`].
    halted: bool,
}

impl Backoff {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a failure and say what to do about it.
    pub fn on_failure(&mut self, kind: FailureKind) -> Retry {
        if kind == FailureKind::Auth {
            self.halted = true;
        }
        if self.halted {
            // Latched: a network failure arriving after a rejected credential
            // must not quietly resume retrying. The credential is still bad.
            return Retry::Never;
        }
        let secs = SCHEDULE_SECS[self.step];
        // Saturate at the last entry rather than wrapping — the cap is the
        // point, and wrapping would send a week-long outage back to 1s.
        self.step = (self.step + 1).min(SCHEDULE_SECS.len() - 1);
        Retry::After(Duration::from_secs(secs))
    }

    /// A pass succeeded: the next failure starts from the top of the schedule
    /// again. Does **not** clear an auth halt — a success cannot happen while
    /// halted, and clearing it here would hide a re-pair that never happened.
    pub fn on_success(&mut self) {
        self.step = 0;
    }

    /// Clear everything, including an auth halt. This is what pairing calls.
    pub fn reset(&mut self) {
        self.step = 0;
        self.halted = false;
    }

    /// Whether a rejected credential has stopped this peer for good.
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// The delay the next failure would yield, for display before it happens.
    /// `None` while halted.
    pub fn next_delay_secs(&self) -> Option<u64> {
        if self.halted {
            None
        } else {
            Some(SCHEDULE_SECS[self.step])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fail(b: &mut Backoff) -> Option<u64> {
        b.on_failure(FailureKind::Unreachable).secs()
    }

    #[test]
    fn schedule_advances_then_caps() {
        let mut b = Backoff::new();
        let observed: Vec<Option<u64>> = (0..9).map(|_| fail(&mut b)).collect();
        assert_eq!(
            observed,
            vec![
                Some(1),
                Some(2),
                Some(5),
                Some(15),
                Some(30),
                Some(60),
                Some(60),
                Some(60),
                Some(60),
            ]
        );
    }

    #[test]
    fn success_resets_the_schedule() {
        let mut b = Backoff::new();
        for _ in 0..4 {
            fail(&mut b);
        }
        assert_eq!(fail(&mut b), Some(30), "should be four steps in");

        b.on_success();
        assert_eq!(fail(&mut b), Some(1), "success must restart at the top");
    }

    #[test]
    fn an_auth_failure_never_retries() {
        let mut b = Backoff::new();
        assert_eq!(b.on_failure(FailureKind::Auth), Retry::Never);
        assert!(b.is_halted());
        assert_eq!(b.next_delay_secs(), None);
    }

    #[test]
    fn an_auth_halt_latches_across_later_network_failures() {
        // The regression this guards: a rejected credential followed by the
        // master going offline must stay `Re-pair required`, not slide back
        // into a retry loop that will never succeed.
        let mut b = Backoff::new();
        b.on_failure(FailureKind::Auth);
        assert_eq!(b.on_failure(FailureKind::Unreachable), Retry::Never);
        assert_eq!(b.on_failure(FailureKind::Unreachable), Retry::Never);
        assert!(b.is_halted());
    }

    #[test]
    fn success_does_not_clear_an_auth_halt_but_reset_does() {
        let mut b = Backoff::new();
        b.on_failure(FailureKind::Auth);
        b.on_success();
        assert!(b.is_halted(), "only re-pairing clears a rejected credential");

        b.reset();
        assert!(!b.is_halted());
        assert_eq!(fail(&mut b), Some(1));
    }

    #[test]
    fn next_delay_previews_without_advancing() {
        let mut b = Backoff::new();
        assert_eq!(b.next_delay_secs(), Some(1));
        assert_eq!(b.next_delay_secs(), Some(1), "peeking must not advance");
        fail(&mut b);
        assert_eq!(b.next_delay_secs(), Some(2));
    }
}
