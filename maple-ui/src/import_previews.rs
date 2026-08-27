//! On-demand thumbnail decoding for the import browser.
//!
//! A card of a few thousand photos will not fit in memory as decoded
//! previews — 196 KB each adds up fast — and almost none of them are on
//! screen. So the browser decodes what the user is *looking at* rather than
//! everything it found.
//!
//! Two ideas carry the whole module:
//!
//! * **Priority is evaluated when a worker picks up work, not when the work
//!   is queued.** The queue holds a set of wanted indices and one `focus`
//!   (where the viewport is); a worker always takes the pending index
//!   nearest that focus. Scrolling therefore re-prioritises everything
//!   already queued by writing a single number — no re-sorting, no
//!   cancelling, and no waiting through a backlog of photos that scrolled
//!   past long ago before the ones now on screen get decoded.
//! * **Reading stays serial.** The medium is one bus (see the scan pipeline
//!   in `import.rs`); several readers on it are slower than one. Workers
//!   here take a read lock in turn and only the decode runs in parallel.
//!
//! The service outlives a scan: the user keeps scrolling long after the
//! index pass has finished, and every one of those scrolls is a new
//! request.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Duration;

/// The files one photo can be previewed from, in preference order.
#[derive(Clone)]
pub struct PhotoRef {
    pub display: PathBuf,
    pub companions: Vec<PathBuf>,
}

/// One canonical preview on its way back to the UI thread.
///
/// It travels as WebP, not as pixels: that is the representation the entry
/// keeps and the one every check reads, so encoding it here — on a worker,
/// beside the decode — rather than on the UI thread is both cheaper and the
/// only way the service and the scan can be said to produce the same thing.
pub enum PreviewMsg {
    Ready {
        index: usize,
        /// The canonical preview (see [`maple_import::preview`]).
        webp: Vec<u8>,
        /// The display file would not decode and a companion stood in — a
        /// corrupt JPEG next to an intact raw.
        from_companion: bool,
    },
    /// Nothing in the group could be decoded. The tile keeps its "no
    /// preview" mark.
    Failed { index: usize },
}

/// Read one photo and encode its canonical preview, falling back to the
/// group's companions.
///
/// A raw + JPEG pair lists the JPEG as the display file, so one corrupt
/// JPEG would otherwise lose the preview for a photo whose raw is intact.
///
/// This is the *recovery* path. The scan already makes a canonical preview
/// for every photo it reads and the medium's cache holds the ones it made
/// on an earlier run, so by the time the strip asks for something here it
/// is a photo the scan has not reached yet, or one whose display file would
/// not decode.
///
/// `reader` is held **only across the reads**, never across the encode: the
/// medium is the part that must be taken in turn, and holding it through
/// the encode would collapse the pool back to one worker.
fn preview_photo(
    photo: &PhotoRef,
    budget: Duration,
    reader: &Mutex<()>,
) -> Option<(Vec<u8>, bool)> {
    let read = |path: &PathBuf| {
        let _turn = reader.lock();
        crate::import::read_preview_bytes(path, budget)
    };

    if let Some(bytes) = read(&photo.display) {
        if let Ok(webp) = maple_import::preview::encode(&bytes) {
            return Some((webp, false));
        }
        tracing::warn!(
            target: "maple::import::unreadable",
            "decode failed: {}",
            photo.display.display()
        );
    }
    for companion in &photo.companions {
        let Some(bytes) = read(companion) else { continue };
        if let Ok(webp) = maple_import::preview::encode(&bytes) {
            tracing::info!(
                target: "maple::import::unreadable",
                "preview recovered from {} because {} would not decode",
                companion.display(),
                photo.display.display()
            );
            return Some((webp, true));
        }
    }
    None
}

/// What still needs decoding, and where the user is looking.
struct Queue {
    /// Indices wanted but not yet handed to a worker.
    pending: HashSet<usize>,
    /// Row the viewport is centred on. Workers decode outwards from here.
    focus: usize,
    stopped: bool,
}

impl Queue {
    /// The pending index nearest the focus, removed from the set.
    ///
    /// Ties break towards the index *after* the focus, which is the
    /// direction a filmstrip is usually being read in.
    fn take_nearest(&mut self) -> Option<usize> {
        let focus = self.focus;
        let best = *self
            .pending
            .iter()
            .min_by_key(|&&i| (i.abs_diff(focus), i < focus))?;
        self.pending.remove(&best);
        Some(best)
    }
}

/// A pool of decode workers fed by whatever the viewport currently wants.
pub struct PreviewService {
    queue: Arc<(Mutex<Queue>, Condvar)>,
    stopping: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl PreviewService {
    /// Start `workers` decode threads over `paths`, indexed by scan index.
    ///
    /// Nothing is decoded until [`Self::want`] is called: an import browser
    /// that is never scrolled should not touch the card at all.
    pub fn start(
        photos: Arc<Vec<PhotoRef>>,
        budget: Duration,
        workers: usize,
        tx: mpsc::Sender<PreviewMsg>,
    ) -> Self {
        let queue = Arc::new((
            Mutex::new(Queue {
                pending: HashSet::new(),
                focus: 0,
                stopped: false,
            }),
            Condvar::new(),
        ));
        let stopping = Arc::new(AtomicBool::new(false));
        // One reader at a time: the decode is what parallelises, not the
        // read. See the module docs.
        let reader = Arc::new(Mutex::new(()));

        let threads = (0..workers.max(1))
            .map(|_| {
                let queue = queue.clone();
                let stopping = stopping.clone();
                let photos = photos.clone();
                let reader = reader.clone();
                let tx = tx.clone();
                std::thread::spawn(move || while let Some(index) = next_index(&queue) {
                    if stopping.load(Ordering::Relaxed) {
                        break;
                    }
                    let Some(photo) = photos.get(index) else { continue };

                    let msg = match preview_photo(photo, budget, &reader) {
                        Some((webp, from_companion)) => {
                            PreviewMsg::Ready { index, webp, from_companion }
                        }
                        None => PreviewMsg::Failed { index },
                    };
                    if tx.send(msg).is_err() {
                        break;
                    }
                })
            })
            .collect();

        Self { queue, stopping, threads }
    }

    /// Replace what the service is working on.
    ///
    /// `wanted` is everything the UI would like decoded right now — the
    /// viewport plus its prefetch margin, minus whatever it already holds.
    /// Replacing rather than appending is what keeps a fast scroll from
    /// accumulating a backlog of photos nobody is looking at any more.
    pub fn want(&self, wanted: impl IntoIterator<Item = usize>, focus: usize) {
        let (lock, cvar) = &*self.queue;
        let Ok(mut queue) = lock.lock() else { return };
        queue.pending = wanted.into_iter().collect();
        queue.focus = focus;
        if !queue.pending.is_empty() {
            cvar.notify_all();
        }
    }

    /// Stop the pool and wait for its threads.
    ///
    /// A worker already inside a read cannot be interrupted, only outlived,
    /// so this waits out at most one read budget per worker.
    pub fn stop(&mut self) {
        self.stopping.store(true, Ordering::Relaxed);
        {
            let (lock, cvar) = &*self.queue;
            if let Ok(mut queue) = lock.lock() {
                queue.stopped = true;
                queue.pending.clear();
            }
            cvar.notify_all();
        }
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for PreviewService {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Block until there is something to decode, or the service is stopping.
fn next_index(queue: &Arc<(Mutex<Queue>, Condvar)>) -> Option<usize> {
    let (lock, cvar) = &**queue;
    let mut guard = lock.lock().ok()?;
    loop {
        if guard.stopped {
            return None;
        }
        if let Some(index) = guard.take_nearest() {
            return Some(index);
        }
        guard = cvar.wait(guard).ok()?;
    }
}

/// Decoded previews held in memory, least-recently-seen first out.
///
/// Recency is "when the user last had it in view", not when it was decoded:
/// scrolling back to a photo puts it at the top again, which is what stops
/// a slow pan from evicting the very tiles it is about to need.
pub struct Retained {
    order: Vec<usize>,
    cap: usize,
}

impl Retained {
    pub fn new(cap: usize) -> Self {
        Self { order: Vec::new(), cap: cap.max(1) }
    }

    /// Mark `index` as just seen. Returns nothing — touching never evicts,
    /// because everything being touched is on screen.
    pub fn touch(&mut self, index: usize) {
        if let Some(pos) = self.order.iter().position(|&i| i == index) {
            self.order.remove(pos);
        }
        self.order.push(index);
    }

    /// Record a newly decoded preview, returning any that must be dropped
    /// to stay under the cap.
    pub fn insert(&mut self, index: usize) -> Vec<usize> {
        self.touch(index);
        let mut evicted = Vec::new();
        while self.order.len() > self.cap {
            evicted.push(self.order.remove(0));
        }
        evicted
    }

    /// How many previews may be held at once.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Forget everything (a new scan).
    pub fn clear(&mut self) {
        self.order.clear();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.order.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue(pending: &[usize], focus: usize) -> Queue {
        Queue {
            pending: pending.iter().copied().collect(),
            focus,
            stopped: false,
        }
    }

    #[test]
    fn the_nearest_pending_photo_to_the_viewport_is_decoded_first() {
        // The whole point: a user who scrolls to photo 500 must not wait
        // for 0..499 to decode first.
        let mut q = queue(&[0, 1, 2, 498, 500, 502], 500);
        assert_eq!(q.take_nearest(), Some(500));
        assert_eq!(q.take_nearest(), Some(502), "ties break forwards");
        assert_eq!(q.take_nearest(), Some(498));
        assert_eq!(q.take_nearest(), Some(2));
    }

    #[test]
    fn an_empty_queue_has_nothing_to_take() {
        let mut q = queue(&[], 0);
        assert_eq!(q.take_nearest(), None);
    }

    #[test]
    fn moving_the_focus_reprioritises_what_is_already_queued() {
        let mut q = queue(&[10, 20, 30], 10);
        assert_eq!(q.take_nearest(), Some(10));
        // The user scrolled away before the rest were picked up.
        q.focus = 30;
        assert_eq!(q.take_nearest(), Some(30));
        assert_eq!(q.take_nearest(), Some(20));
    }

    #[test]
    fn retention_drops_the_least_recently_seen() {
        let mut r = Retained::new(3);
        for i in 0..3 {
            assert!(r.insert(i).is_empty());
        }
        assert_eq!(r.insert(3), vec![0], "the oldest goes first");
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn seeing_a_preview_again_moves_it_back_to_the_top() {
        let mut r = Retained::new(3);
        for i in 0..3 {
            r.insert(i);
        }
        // Scrolling back to 0 must save it from the next eviction.
        r.touch(0);
        assert_eq!(r.insert(3), vec![1]);
        assert_eq!(r.insert(4), vec![2]);
        assert_eq!(r.insert(5), vec![0], "and only then the one we revisited");
    }

    #[test]
    fn touching_never_evicts() {
        let mut r = Retained::new(2);
        r.insert(1);
        r.insert(2);
        r.touch(1);
        r.touch(2);
        assert_eq!(r.len(), 2, "everything on screen stays on screen");
    }

    #[test]
    fn a_cap_below_a_screenful_is_raised_rather_than_thrashing() {
        let r = Retained::new(0);
        assert_eq!(r.capacity(), 1);
    }

    // ── Decoding, and falling back to the raw ────────────────────

    fn write_png(path: &std::path::Path) {
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([120, 30, 200]));
        image::DynamicImage::ImageRgb8(img).save(path).unwrap();
    }

    /// Bytes that look like a JPEG for long enough to be picked up, and
    /// then are not one.
    fn write_corrupt_jpeg(path: &std::path::Path) {
        std::fs::write(path, b"\xff\xd8\xff\xe0 not actually a jpeg").unwrap();
    }

    /// Preview one photo and decode the result back, so a test can assert
    /// on pixels while the service still returns only the canonical WebP.
    fn preview(display: PathBuf, companions: Vec<PathBuf>) -> Option<(Vec<u8>, u32, u32, bool)> {
        let photo = PhotoRef { display, companions };
        let (webp, from_companion) =
            preview_photo(&photo, Duration::from_secs(5), &Mutex::new(()))?;
        let frame = maple_import::preview::decode(&webp).expect("canonical preview must decode");
        let (w, h) = frame.dimensions();
        Some((frame.into_raw(), w, h, from_companion))
    }

    #[test]
    fn a_readable_photo_decodes_from_its_own_file() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("a.png");
        write_png(&png);

        let (rgb, w, h, from_companion) = preview(png, vec![]).unwrap();
        assert!(!rgb.is_empty() && w > 0 && h > 0);
        assert!(!from_companion);
    }

    #[test]
    fn a_corrupt_display_file_is_previewed_from_its_companion() {
        // The shape of a RAW+JPEG pair: the JPEG is the display file, and
        // it is the one that is corrupt.
        let dir = tempfile::tempdir().unwrap();
        let jpg = dir.path().join("DSCF0042.jpg");
        write_corrupt_jpeg(&jpg);
        let companion = dir.path().join("DSCF0042-raw.png");
        write_png(&companion);

        let (rgb, w, _, from_companion) = preview(jpg, vec![companion]).unwrap();
        assert!(from_companion, "the companion should have stood in");
        assert!(w > 0 && !rgb.is_empty());
    }

    #[test]
    fn one_unusable_companion_does_not_stop_us_trying_the_next() {
        let dir = tempfile::tempdir().unwrap();
        let jpg = dir.path().join("DSCF0043.jpg");
        write_corrupt_jpeg(&jpg);
        let junk = dir.path().join("DSCF0043-junk.png");
        std::fs::write(&junk, b"also not an image").unwrap();
        let good = dir.path().join("DSCF0043-raw.png");
        write_png(&good);

        let (_, _, _, from_companion) = preview(jpg, vec![junk, good]).unwrap();
        assert!(from_companion);
    }

    #[test]
    fn a_corrupt_photo_with_nothing_to_fall_back_on_has_no_preview() {
        let dir = tempfile::tempdir().unwrap();
        let jpg = dir.path().join("DSCF0044.jpg");
        write_corrupt_jpeg(&jpg);

        assert!(preview(jpg, vec![]).is_none());
    }

    // ── The service end to end ───────────────────────────────────

    #[test]
    fn only_what_was_asked_for_is_decoded() {
        let dir = tempfile::tempdir().unwrap();
        let photos: Vec<PhotoRef> = (0..10)
            .map(|i| {
                let path = dir.path().join(format!("p{i}.png"));
                write_png(&path);
                PhotoRef { display: path, companions: vec![] }
            })
            .collect();

        let (tx, rx) = mpsc::channel();
        let service = PreviewService::start(Arc::new(photos), Duration::from_secs(5), 2, tx);
        service.want([3, 4, 5], 4);

        let mut decoded = HashSet::new();
        for _ in 0..3 {
            match rx.recv_timeout(Duration::from_secs(10)).expect("preview") {
                PreviewMsg::Ready { index, webp, .. } => {
                    assert!(maple_import::preview::decode(&webp).is_ok());
                    decoded.insert(index);
                }
                PreviewMsg::Failed { index } => panic!("photo {index} failed to decode"),
            }
        }
        assert_eq!(decoded, HashSet::from([3, 4, 5]));

        // Nothing else was touched: an import browser that is never
        // scrolled must not decode a whole card.
        drop(service);
        assert!(rx.try_recv().is_err(), "decoded a photo nobody asked for");
    }

    #[test]
    fn a_photo_that_cannot_be_decoded_comes_back_as_failed() {
        let dir = tempfile::tempdir().unwrap();
        let jpg = dir.path().join("broken.jpg");
        write_corrupt_jpeg(&jpg);

        let (tx, rx) = mpsc::channel();
        let service = PreviewService::start(
            Arc::new(vec![PhotoRef { display: jpg, companions: vec![] }]),
            Duration::from_secs(5),
            1,
            tx,
        );
        service.want([0], 0);

        match rx.recv_timeout(Duration::from_secs(10)).expect("a verdict") {
            PreviewMsg::Failed { index } => assert_eq!(index, 0),
            PreviewMsg::Ready { .. } => panic!("corrupt bytes should not decode"),
        }
    }
}
