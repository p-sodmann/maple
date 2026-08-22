//! Which photos are missing their bytes, and what it means to acquire them.
//!
//! §3.5 gave a row two independent facts: `status` says whether the library
//! entry is usable, and `locality` says whether *this* machine holds the
//! file. P6 made a `remote` row perfectly browsable by fetching its pixels on
//! demand. P7 is the other half — a **full** or **partial** peer wants those
//! bytes on disk, and the three queries here are how each side works out what
//! it is short of.
//!
//! # Content hashes, not row ids
//!
//! Every route in this direction is keyed by BLAKE3. Two devices that both
//! hold a photo name it identically without ever agreeing on a rowid, a
//! transfer that is interrupted resumes by asking for the same hash again,
//! and — the part that matters for a network service — the receiver can
//! *verify* what arrived. A row id could not be checked against anything.
//!
//! # Nothing here stamps
//!
//! `path`, `raw_path`, `filename` and `locality` are machine-local, so
//! acquiring a file changes nothing any peer should hear about: the photo is
//! the same photo, and where its bytes sit is this device's business. See
//! the writer rules in `CLAUDE.md`.

use std::path::{Path, PathBuf};

use rusqlite::{params, OptionalExtension};

use crate::{path_to_db, Database};

/// A library row whose bytes live on another device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingOriginal {
    pub id: i64,
    /// Content hash — the blob route's key, and what the receiver checks the
    /// downloaded bytes against.
    pub hash: [u8; 32],
    /// The name the origin device knows this file by. Supplies the
    /// `{original}` token and the extension when the file is placed, so a
    /// photo that arrives over the wire is named the way the same photo would
    /// be if it had been imported from the card here.
    pub filename: String,
    /// The origin's name for the companion raw file, when it has one.
    pub raw_filename: Option<String>,
    /// What the origin says the file weighs. Advisory — used to log and to
    /// estimate, never trusted enough to skip verifying the bytes.
    pub file_size: i64,
}

impl Database {
    /// Photos this device lists but does not hold, oldest row first.
    ///
    /// `ORDER BY id` rather than anything cleverer: a transfer runs in
    /// batches across many passes, and a stable order means the second pass
    /// starts where the first stopped instead of re-shuffling the queue.
    ///
    /// A row whose content this library already holds under a *different*
    /// row is not filtered out. It could be — the bytes are on the disk
    /// already — but the two rows are separate library entries with separate
    /// guids, and quietly pointing both at one file would make deleting
    /// either one delete the other's photo. Importing the same card twice
    /// copies twice for the same reason.
    pub fn originals_to_fetch(&self, limit: usize) -> anyhow::Result<Vec<MissingOriginal>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, hash, filename, raw_path, file_size
             FROM images
             WHERE locality = 'remote' AND status = 'present'
             ORDER BY id
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |r| {
            Ok(MissingOriginal {
                id: r.get(0)?,
                hash: hash_from(r.get::<_, Vec<u8>>(1)?),
                filename: r.get(2)?,
                // The origin's *path*; only its final component is a name we
                // can use here, and the rest describes a disk we cannot see.
                raw_filename: r.get::<_, Option<String>>(3)?.as_deref().map(basename),
                file_size: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// The hashes this device is missing, for a peer to offer to fill in.
    ///
    /// This is the master's half of a **partial** or **full** link. The
    /// master never dials anybody — a servant is behind whatever NAT it is
    /// behind, and only it knows how to reach the master — so "master fetches
    /// the servant's originals" (§3.8) has to happen as the servant asking
    /// what is wanted and uploading it. Answering this question is the whole
    /// of the master's participation.
    ///
    /// Distinct, because several rows can share a hash and one upload
    /// satisfies all of them.
    pub fn wanted_hashes(&self, limit: usize) -> anyhow::Result<Vec<[u8; 32]>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT hash FROM images
             WHERE locality = 'remote' AND status = 'present'
             ORDER BY id
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |r| {
            Ok(hash_from(r.get::<_, Vec<u8>>(0)?))
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// The lowest-id row waiting for this hash, or `None` if nothing here
    /// wants it.
    ///
    /// The upload route's admission check, and the reason a paired peer
    /// cannot use it to write arbitrary files into this library: it can only
    /// ever supply bytes for a row this device already replicated the
    /// metadata of, and the hash of what it sends has to match what that row
    /// already says.
    pub fn row_wanting(&self, hash: &[u8; 32]) -> anyhow::Result<Option<MissingOriginal>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, hash, filename, raw_path, file_size
                 FROM images
                 WHERE hash = ?1 AND locality = 'remote' AND status = 'present'
                 ORDER BY id LIMIT 1",
                params![hash.as_slice()],
                |r| {
                    Ok(MissingOriginal {
                        id: r.get(0)?,
                        hash: hash_from(r.get::<_, Vec<u8>>(1)?),
                        filename: r.get(2)?,
                        raw_filename: r.get::<_, Option<String>>(3)?.as_deref().map(basename),
                        file_size: r.get(4)?,
                    })
                },
            )
            .optional()?)
    }

    /// Record that this device now holds the file for `id`, at `path`.
    ///
    /// Every path column is rewritten together, because after this call they
    /// all have to describe *this* disk: `path` and `raw_path` were the
    /// origin's until now, and `filename` was the origin's name for a file
    /// that has just been renamed by the local path template. `raw_path` goes
    /// to whatever companion was actually placed — `None` clears the origin's,
    /// rather than leaving a pointer to a file on another machine.
    ///
    /// Not stamped, and deliberately: see the module docs.
    pub fn adopt_original(
        &self,
        id: i64,
        path: &Path,
        raw_path: Option<&Path>,
    ) -> anyhow::Result<()> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        self.conn.execute(
            "UPDATE images
             SET path = ?1, raw_path = ?2, filename = ?3, locality = 'local', status = 'present'
             WHERE id = ?4",
            params![path_to_db(path), raw_path.map(path_to_db), filename, id],
        )?;
        Ok(())
    }

    /// Where this device keeps the file for `hash`, if it keeps it at all.
    ///
    /// The upload side of the same question [`Database::blob_path`] answers
    /// for serving: a photo already on this disk needs no transfer, in either
    /// direction.
    pub fn holds_original(&self, hash: &[u8; 32]) -> anyhow::Result<Option<PathBuf>> {
        self.blob_path(hash, false)
    }
}

/// The final component of a path written by *another* device, which may use
/// the other separator. Splitting on both is what makes a Windows master's
/// `C:\photos\a.raf` land as `a.raf` on a Linux servant.
fn basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_owned()
}

fn hash_from(blob: Vec<u8>) -> [u8; 32] {
    let mut hash = [0u8; 32];
    if blob.len() == 32 {
        hash.copy_from_slice(&blob);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A library holding one photo of its own and two relayed from a peer,
    /// one of which has a raw companion on that peer.
    fn library() -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(&dir.path().join("library.db")).expect("open");

        let mine = dir.path().join("mine.jpg");
        std::fs::write(&mine, b"local bytes").expect("write");
        db.insert_image(&mine, &[0x11u8; 32], 11).expect("insert");

        for (guid, hash, path, raw) in [
            ("g-theirs", [0x22u8; 32], "/workstation/photos/theirs.jpg", None),
            (
                "g-raw",
                [0x33u8; 32],
                "/workstation/photos/DSCF0001.JPG",
                Some("/workstation/photos/DSCF0001.RAF"),
            ),
        ] {
            db.conn
                .execute(
                    "INSERT INTO images(path, hash, file_size, added_at, status, filename,
                                        raw_path, guid, rev, rev_dev, locality, origin_device)
                     VALUES (?1, ?2, 99, 100, 'present', ?3, ?4, ?5, 1, 'dev-master',
                             'remote', 'dev-master')",
                    rusqlite::params![
                        path,
                        hash.as_slice(),
                        basename(path),
                        raw,
                        guid
                    ],
                )
                .expect("insert remote");
        }
        (dir, db)
    }

    #[test]
    fn only_relayed_rows_are_queued_for_fetching() {
        let (_dir, db) = library();
        let queued = db.originals_to_fetch(10).unwrap();

        assert_eq!(queued.len(), 2, "the local photo is not missing anything");
        assert_eq!(queued[0].filename, "theirs.jpg");
        assert_eq!(queued[0].raw_filename, None);
        assert_eq!(queued[0].hash, [0x22u8; 32]);
        // The origin's *directories* are none of this device's business; only
        // the name survives, to be re-filed under the local path template.
        assert_eq!(queued[1].raw_filename.as_deref(), Some("DSCF0001.RAF"));
    }

    #[test]
    fn the_queue_is_stable_and_respects_its_limit() {
        // A transfer runs across many passes; an order that varied between
        // them would re-shuffle the queue instead of continuing it.
        let (_dir, db) = library();
        for _ in 0..3 {
            let first = db.originals_to_fetch(1).unwrap();
            assert_eq!(first.len(), 1);
            assert_eq!(first[0].filename, "theirs.jpg");
        }
    }

    #[test]
    fn wanted_hashes_are_what_this_device_lacks() {
        let (_dir, db) = library();
        let wanted = db.wanted_hashes(10).unwrap();
        assert_eq!(wanted, vec![[0x22u8; 32], [0x33u8; 32]]);
        assert!(
            !wanted.contains(&[0x11u8; 32]),
            "a photo already on this disk is not wanted"
        );
    }

    #[test]
    fn an_unwanted_hash_has_no_row_to_land_in() {
        // This is the upload route's admission check: a paired peer can only
        // supply bytes for a row this device already replicated.
        let (_dir, db) = library();
        assert!(db.row_wanting(&[0x22u8; 32]).unwrap().is_some());
        assert!(
            db.row_wanting(&[0x11u8; 32]).unwrap().is_none(),
            "already local"
        );
        assert!(db.row_wanting(&[0x99u8; 32]).unwrap().is_none(), "unknown");
    }

    #[test]
    fn adopting_rewrites_every_path_column_to_describe_this_disk() {
        let (dir, db) = library();
        let row = db.row_wanting(&[0x33u8; 32]).unwrap().expect("queued");
        let placed = dir.path().join("2024/03/20240315_DSCF0001.JPG");
        let raw = dir.path().join("2024/03/20240315_DSCF0001.RAF");

        db.adopt_original(row.id, &placed, Some(&raw)).unwrap();

        let (path, raw_path, filename, locality): (String, Option<String>, String, String) = db
            .conn
            .query_row(
                "SELECT path, raw_path, filename, locality FROM images WHERE id = ?1",
                rusqlite::params![row.id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .unwrap();
        assert!(path.ends_with("20240315_DSCF0001.JPG"), "{path}");
        assert!(raw_path.unwrap().ends_with("20240315_DSCF0001.RAF"));
        assert_eq!(
            filename, "20240315_DSCF0001.JPG",
            "the displayed name follows the file, not the origin"
        );
        assert_eq!(locality, "local");
        assert!(db.row_wanting(&[0x33u8; 32]).unwrap().is_none());
        // And it now serves: `blob_path` skips remote rows, so this is the
        // proof that a servant which fetched a photo can relay it onward.
        assert!(db.holds_original(&[0x33u8; 32]).unwrap().is_some());
    }

    #[test]
    fn adopting_without_a_companion_clears_the_origins_raw_path() {
        // Otherwise the row keeps pointing at a RAF on another machine, and
        // every later reader has to know that column is sometimes a lie.
        let (dir, db) = library();
        let row = db.row_wanting(&[0x33u8; 32]).unwrap().expect("queued");
        db.adopt_original(row.id, &dir.path().join("a.jpg"), None).unwrap();

        let raw_path: Option<String> = db
            .conn
            .query_row(
                "SELECT raw_path FROM images WHERE id = ?1",
                rusqlite::params![row.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(raw_path, None);
    }

    #[test]
    fn adopting_does_not_stamp() {
        // `path`, `raw_path`, `filename` and `locality` describe this disk,
        // not the photo. Shipping a rev for them would have every device tell
        // the others where *its* copy lives.
        let (dir, db) = library();
        let row = db.row_wanting(&[0x22u8; 32]).unwrap().expect("queued");
        let before: (i64, String) = db
            .conn
            .query_row(
                "SELECT rev, rev_dev FROM images WHERE id = ?1",
                rusqlite::params![row.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();

        db.adopt_original(row.id, &dir.path().join("a.jpg"), None).unwrap();

        let after: (i64, String) = db
            .conn
            .query_row(
                "SELECT rev, rev_dev FROM images WHERE id = ?1",
                rusqlite::params![row.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn a_windows_masters_backslashes_still_yield_a_filename() {
        assert_eq!(basename(r"C:\photos\2024\DSCF0001.RAF"), "DSCF0001.RAF");
        assert_eq!(basename("/photos/a.jpg"), "a.jpg");
        assert_eq!(basename("bare.jpg"), "bare.jpg");
    }
}
