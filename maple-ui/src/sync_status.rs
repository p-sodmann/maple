//! The sidebar status pill: a 1-second poller over the shared status cell.
//!
//! # Why a poller and not a callback
//!
//! The sync worker runs off the UI thread and cannot touch Slint properties.
//! It could post each transition through `upgrade_in_event_loop`, but a pill
//! wants the *current* level and nothing else — see `maple_sync::status` for
//! why that argues for a shared cell over a channel. A poller reading that
//! cell keeps the worker free of any UI dependency at all, which matters
//! because P5 spawns it from a plain `std::thread`.
//!
//! # Why the timer is held
//!
//! `slint::Timer` stops the moment it is dropped, so a timer created inside
//! `run()` and left unbound would fire zero times. [`PillHandle`] keeps it
//! alive for the life of the window, the same idiom `path_template_window`
//! uses for its restructure poller.

use std::cell::Cell;
use std::rc::Rc;
use std::time::Duration;

use slint::{ComponentHandle, SharedString, Timer, TimerMode};

use maple_sync::{StatusCell, SyncState, SyncStatus};

use crate::sync_supervisor::SyncSupervisor;
use crate::AppWindow;

/// How often the pill re-reads the status cell.
const POLL: Duration = Duration::from_secs(1);

/// Keeps the poller alive. Dropping this stops the pill updating.
pub struct PillHandle {
    _timer: Timer,
}

/// Start the poller and paint the pill once immediately, so the sidebar is
/// never blank for the first second after launch.
pub fn wire(
    window: &AppWindow,
    status: StatusCell,
    sync: Rc<SyncSupervisor>,
    now_ms: impl Fn() -> i64 + 'static,
) -> PillHandle {
    // "Retry" beside an offline pill. Shown only where it can do something —
    // see `StatusDisplay::retryable`, which is also what keeps it off the
    // `Unauthorized` state, the one that most looks like it wants it.
    window.on_sync_retry_clicked({
        let w = window.as_weak();
        let status = status.clone();
        let sync = sync.clone();
        move || {
            if !sync.retry_now() {
                // Nothing to wake. Rather than a button that does nothing,
                // fall back to what the pill does: the user's next move is in
                // Settings either way.
                tracing::info!("sync: no worker to retry; opening settings instead");
                if let Some(w) = w.upgrade() {
                    w.invoke_settings_clicked();
                }
                return;
            }
            // Answer the click before the worker can: a pass opens with a
            // `hello` round trip, and a button that leaves the pill red for
            // two seconds reads as a button that did not work. The worker
            // overwrites this with `Running` or `Offline` shortly.
            acknowledge(&status);
            if let Some(w) = w.upgrade() {
                apply(&w, &status, now_ms_once());
            }
        }
    });

    window.on_sync_pill_clicked({
        // Opening Settings is all a click does for now. §1.3 asks for it to
        // scroll to the Sync card too; the card lives inside a ScrollView
        // with no addressable anchor, so that is left for whoever gives the
        // settings body real sections.
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                w.invoke_settings_clicked();
            }
        }
    });

    apply(window, &status, now_ms());

    // Toggled on every tick to drive the dot's pulse; see `SyncPill`.
    let pulse = Rc::new(Cell::new(false));
    let timer = Timer::default();
    timer.start(TimerMode::Repeated, POLL, {
        let w = window.as_weak();
        let status = status.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            pulse.set(!pulse.get());
            w.set_sync_pill_pulse_on(pulse.get());
            apply(&w, &status, now_ms());
        }
    });

    PillHandle { _timer: timer }
}

/// Read the cell once and push the result into the window's properties.
fn apply(window: &AppWindow, status: &StatusCell, now_ms: i64) {
    // A poisoned mutex means the worker panicked mid-update. The status it
    // left behind is still the last thing that was true, and showing it beats
    // taking the UI down with the worker.
    let snapshot: SyncStatus = match status.lock() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };
    let display = snapshot.display();
    let (r, g, b) = display.tone.rgb();

    window.set_sync_pill_label(SharedString::from(display.label));
    window.set_sync_pill_color(slint::Color::from_rgb_u8(r, g, b));
    window.set_sync_pill_pulsing(display.pulsing);
    window.set_sync_pill_can_retry(display.retryable);
    window.set_sync_pill_tooltip(SharedString::from(snapshot.tooltip(now_ms)));
}

/// Move the pill off `Offline` the moment the user asks for a retry.
///
/// Only the `state` field, not a fresh [`SyncStatus`]: `for_role` would also
/// throw away `last_sync_ms` and the pending count, and the tooltip would go
/// blank for the second before the worker's first status write.
fn acknowledge(status: &StatusCell) {
    let mut guard = match status.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.state = SyncState::Connecting;
    guard.last_error = None;
}

/// The clock for the one repaint that does not happen on a tick.
fn now_ms_once() -> i64 {
    maple_sync::now_ms()
}

/// Overwrite the shared status in place.
///
/// Used by the settings card when the user changes this device's role — the
/// pill must follow immediately rather than waiting for a worker that, before
/// P5, does not exist.
pub fn set_status(cell: &StatusCell, status: SyncStatus) {
    match cell.lock() {
        Ok(mut guard) => *guard = status,
        Err(poisoned) => *poisoned.into_inner() = status,
    }
}
