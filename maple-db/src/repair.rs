//! One-off repairs for libraries damaged by a bug that has since been fixed.
//!
//! Not migrations. A `PRAGMA user_version` step replays on every fresh
//! database and may only touch SQL; what is wanted here moves *files*, is
//! destructive, and applies to exactly the installations that ran one broken
//! version. So it lives behind a binary the user runs deliberately
//! (`cargo run -p maple-db --bin repair-companions`), reports by default and
//! changes nothing without `--apply`.
//!
//! # Split companions
//!
//! Until `maple_import::place_pair`, a synced photo's companion raw was filed
//! by re-deriving the path template from a blob staged under a synthetic
//! `<hash>.raw` name. Nothing could tell that blob was a raw container, so it
//! parsed as having no EXIF, fell back to the file's mtime, and landed under
//! the month it *arrived* while its display file went under the month it was
//! taken. The `images` row still named both files; the disk had them in
//! different folders.
//!
//! That matters because the library scanner regroups from disk, keyed on
//! `(directory, lowercased stem)` — so an orphaned RAF is not a companion at
//! all but a photograph no row claims, and the next 60-second scan inserted a
//! second `images` row for it, stamped it, and replicated the ghost to every
//! peer.
//!
//! Repairing therefore has two halves, and the order between them is
//! load-bearing: the ghost row is found *by* the companion's current path, so
//! it has to be identified before the file moves out from under it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rusqlite::params;

use crate::{path_from_db, path_to_db, Database};

/// One photo whose companion is not beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitCompanion {
    /// The `images` row that names both files.
    pub image_id: i64,
    /// Where the display file is.
    pub display: PathBuf,
    /// Where the companion currently is.
    pub raw: PathBuf,
    /// Where the companion belongs: beside `display`, same stem, its own
    /// extension. `None` when that name is already taken by an unrelated
    /// file, which this repair refuses to resolve on its own.
    pub belongs_at: Option<PathBuf>,
    /// Rows the scanner minted for the orphan — `images` rows whose own
    /// `path` is `raw`. Normally one; more than one is impossible
    /// (`images.path` is UNIQUE) but the plural costs nothing.
    pub ghosts: Vec<i64>,
}

impl SplitCompanion {
    /// Whether this one can be repaired without a human deciding something.
    pub fn actionable(&self) -> bool {
        self.belongs_at.is_some()
    }
}

/// What a repair pass did, or would do.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RepairReport {
    pub moved: usize,
    pub ghosts_deleted: usize,
    /// Split pairs left alone because the companion's proper name is taken by
    /// something else.
    pub blocked: usize,
}

impl Database {
    /// Find every local photo whose companion raw is not beside it.
    ///
    /// "Beside it" is exactly what `maple_import::scan_grouped` means by it:
    /// same parent directory, same stem ignoring case. A companion that
    /// satisfies that is invisible to this repair however odd it looks —
    /// the scanner will group it, which is the only property that matters.
    ///
    /// `locality = 'local'`, because a relayed row's paths name another
    /// machine's disk and there is nothing here to move.
    pub fn split_companions(&self) -> anyhow::Result<Vec<SplitCompanion>> {
        // Every local row's path, so a ghost can be recognised by the one
        // thing that identifies it: its `path` *is* the orphaned companion.
        let mut by_path: HashMap<PathBuf, i64> = HashMap::new();
        {
            let mut stmt = self
                .conn
                .prepare("SELECT id, path FROM images WHERE locality = 'local'")?;
            let rows = stmt.query_map([], |r| {
                Ok((r.get::<_, i64>(0)?, path_from_db(r.get::<_, String>(1)?)))
            })?;
            for row in rows {
                let (id, path) = row?;
                by_path.insert(path, id);
            }
        }

        let mut stmt = self.conn.prepare(
            "SELECT id, path, raw_path FROM images
             WHERE locality = 'local' AND raw_path IS NOT NULL
             ORDER BY id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                path_from_db(r.get::<_, String>(1)?),
                path_from_db(r.get::<_, String>(2)?),
            ))
        })?;

        let mut split = Vec::new();
        for row in rows {
            let (image_id, display, raw) = row?;
            if groups_together(&display, &raw) {
                continue;
            }
            let belongs_at = companion_home(&display, &raw);
            let ghosts = by_path
                .get(&raw)
                .copied()
                .filter(|id| *id != image_id)
                .into_iter()
                .collect();
            split.push(SplitCompanion {
                image_id,
                display,
                raw,
                belongs_at,
                ghosts,
            });
        }
        Ok(split)
    }

    /// Move each split companion beside its display file and delete the rows
    /// the scanner minted for it.
    ///
    /// Per split pair, in this order: delete the ghost rows (tombstoned
    /// first, so the delete *replicates* instead of a peer resurrecting it on
    /// the next pull), then move the file, then repoint `raw_path`. Deleting
    /// first is what keeps `images.path` UNIQUE from being the thing that
    /// fails — a ghost still holds the companion's old path, and while it
    /// does nothing else may claim it.
    ///
    /// `raw_path` is machine-local, so the update does **not** stamp; the
    /// ghost delete does, because a deletion is not local news. See the
    /// writer rules in `CLAUDE.md`.
    ///
    /// A pair whose companion cannot be given its proper name is counted in
    /// [`RepairReport::blocked`] and otherwise left exactly as it was.
    pub fn repair_split_companions(&self) -> anyhow::Result<RepairReport> {
        let mut report = RepairReport::default();
        for split in self.split_companions()? {
            let Some(home) = split.belongs_at.clone() else {
                report.blocked += 1;
                tracing::warn!(
                    "repair: the name beside {} is taken; leaving {} where it is",
                    split.display.display(),
                    split.raw.display()
                );
                continue;
            };

            if !split.ghosts.is_empty() {
                self.tombstone("images", &split.ghosts)?;
                for ghost in &split.ghosts {
                    self.conn
                        .execute("DELETE FROM images WHERE id = ?1", params![ghost])?;
                    report.ghosts_deleted += 1;
                }
            }

            if split.raw.exists() {
                if let Some(parent) = home.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                move_file(&split.raw, &home)?;
                report.moved += 1;
            } else {
                // The file is already gone; the row still points at it. Fixing
                // the pointer is still right — the scanner will mark the photo
                // however it likes, but it will never again read this column
                // as naming a photograph of its own.
                tracing::warn!("repair: {} is missing", split.raw.display());
            }

            // Machine-local: no stamp. See the module docs.
            self.conn.execute(
                "UPDATE images SET raw_path = ?1 WHERE id = ?2",
                params![path_to_db(&home), split.image_id],
            )?;
            tracing::info!(
                "repair: {} → {}",
                split.raw.display(),
                home.display()
            );
        }
        Ok(report)
    }

}

/// The scanner's own grouping key: same parent, same stem ignoring case.
fn groups_together(display: &Path, raw: &Path) -> bool {
    let stem = |p: &Path| {
        p.file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
    };
    display.parent() == raw.parent() && stem(display) == stem(raw) && stem(display).is_some()
}

/// Where a companion belongs, or `None` if that name is already spoken for.
///
/// Deliberately not a `_1` suffix search. A suffix would move the companion
/// away from its display file's stem, which is the very thing being repaired;
/// if the proper name is taken, a human has to look.
fn companion_home(display: &Path, raw: &Path) -> Option<PathBuf> {
    let dir = display.parent()?;
    let stem = display.file_stem()?;
    let mut name = stem.to_owned();
    if let Some(ext) = raw.extension() {
        name.push(".");
        name.push(ext);
    }
    let home = dir.join(name);
    (home != *display && !home.exists()).then_some(home)
}

/// Rename, falling back to copy-and-remove across filesystems.
fn move_file(from: &Path, to: &Path) -> anyhow::Result<()> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to)
        .map_err(|e| anyhow::anyhow!("failed to move {} to {}: {e}", from.display(), to.display()))?;
    if let Err(e) = std::fs::remove_file(from) {
        tracing::warn!("repair: could not remove {}: {e}", from.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_exists(db: &Database, id: i64) -> bool {
        use rusqlite::OptionalExtension;
        db.conn
            .query_row("SELECT 1 FROM images WHERE id = ?1", params![id], |_| Ok(()))
            .optional()
            .unwrap()
            .is_some()
    }

    fn raw_path_of(db: &Database, id: i64) -> Option<PathBuf> {
        db.conn
            .query_row("SELECT raw_path FROM images WHERE id = ?1", params![id], |r| {
                r.get::<_, Option<String>>(0)
            })
            .unwrap()
            .map(path_from_db)
    }

    /// A library holding one photo filed the way the bug filed it: the JPEG
    /// under its capture month, the RAF under the month it arrived, plus the
    /// ghost row the scanner minted for that orphan.
    fn diverged() -> (tempfile::TempDir, Database, i64, i64) {
        let dir = tempfile::tempdir().expect("tempdir");
        let lib = dir.path().join("library");
        std::fs::create_dir_all(lib.join("2019/07")).expect("month");
        std::fs::create_dir_all(lib.join("2026/08")).expect("other month");
        let display = lib.join("2019/07/DSCF0001.JPG");
        let raw = lib.join("2026/08/DSCF0001.RAF");
        std::fs::write(&display, b"display").expect("write");
        std::fs::write(&raw, b"negative").expect("write");

        let db = Database::open(&dir.path().join("library.db")).expect("open");
        db.insert_image_with_raw(&display, &[0x11u8; 32], 7, Some(&raw))
            .expect("insert");
        // What the scanner did on its next pass: the lone RAF is its own
        // group, and nothing claimed its path.
        db.insert_image(&raw, &[0x22u8; 32], 8).expect("ghost");

        let photo = db.image_id_for_path(&display).expect("id").expect("photo row");
        let ghost = db.image_id_for_path(&raw).expect("id").expect("ghost row");
        (dir, db, photo, ghost)
    }

    #[test]
    fn a_split_pair_is_found_with_its_ghost() {
        let (dir, db, photo, ghost) = diverged();
        let split = db.split_companions().unwrap();
        assert_eq!(split.len(), 1);
        assert_eq!(split[0].image_id, photo);
        assert_eq!(split[0].ghosts, vec![ghost]);
        assert_eq!(
            split[0].belongs_at.as_deref(),
            Some(dir.path().join("library/2019/07/DSCF0001.RAF").as_path())
        );
    }

    #[test]
    fn repairing_moves_the_companion_and_removes_the_ghost() {
        let (dir, db, photo, ghost) = diverged();
        let report = db.repair_split_companions().unwrap();
        assert_eq!(report.moved, 1);
        assert_eq!(report.ghosts_deleted, 1);
        assert_eq!(report.blocked, 0);

        let home = dir.path().join("library/2019/07/DSCF0001.RAF");
        assert_eq!(std::fs::read(&home).unwrap(), b"negative");
        assert!(!dir.path().join("library/2026/08/DSCF0001.RAF").exists());
        assert_eq!(raw_path_of(&db, photo).as_deref(), Some(home.as_path()));
        assert!(!image_exists(&db, ghost), "the ghost row is gone");
        assert!(image_exists(&db, photo), "the real photo is not");

        // Idempotent: the pair now groups, so a second pass sees nothing.
        assert!(db.split_companions().unwrap().is_empty());
        assert_eq!(db.repair_split_companions().unwrap(), RepairReport::default());
    }

    #[test]
    fn the_ghost_is_tombstoned_so_the_delete_replicates() {
        // Without this the peer hands the row straight back on the next pull
        // and the ghost reappears — the repair would have to be run forever.
        let (_dir, db, _photo, ghost) = diverged();
        let guid: String = db
            .conn
            .query_row("SELECT guid FROM images WHERE id = ?1", params![ghost], |r| {
                r.get(0)
            })
            .unwrap();
        db.repair_split_companions().unwrap();

        let tombstoned: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sync_tombstones WHERE guid = ?1 AND entity = 'images'",
                params![guid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(tombstoned, 1);
    }

    #[test]
    fn a_pair_that_already_groups_is_left_alone() {
        // Including the case the scanner tolerates and a strict comparison
        // would not: the stem matches apart from case.
        let dir = tempfile::tempdir().unwrap();
        let lib = dir.path().join("library");
        std::fs::create_dir_all(&lib).unwrap();
        let display = lib.join("DSCF0001.JPG");
        let raw = lib.join("dscf0001.raf");
        std::fs::write(&display, b"display").unwrap();
        std::fs::write(&raw, b"negative").unwrap();
        let db = Database::open(&dir.path().join("library.db")).unwrap();
        db.insert_image_with_raw(&display, &[0x11u8; 32], 7, Some(&raw))
            .unwrap();

        assert!(db.split_companions().unwrap().is_empty());
    }

    #[test]
    fn a_companion_whose_name_is_taken_is_reported_not_guessed() {
        let (dir, db, photo, _ghost) = diverged();
        // Something unrelated already sits where the companion belongs.
        std::fs::write(dir.path().join("library/2019/07/DSCF0001.RAF"), b"someone else")
            .unwrap();

        let split = db.split_companions().unwrap();
        assert_eq!(split.len(), 1);
        assert!(!split[0].actionable());

        let report = db.repair_split_companions().unwrap();
        assert_eq!(report, RepairReport { moved: 0, ghosts_deleted: 0, blocked: 1 });
        // Nothing was touched: a suffixed name would move the companion off
        // its display file's stem, which is the very thing being repaired.
        assert_eq!(
            std::fs::read(dir.path().join("library/2019/07/DSCF0001.RAF")).unwrap(),
            b"someone else"
        );
        assert!(dir.path().join("library/2026/08/DSCF0001.RAF").exists());
        assert!(raw_path_of(&db, photo)
            .unwrap()
            .ends_with("2026/08/DSCF0001.RAF"));
    }

    #[test]
    fn a_split_pair_with_no_ghost_is_still_repaired() {
        // A library repaired before its scanner got round to the orphan.
        let dir = tempfile::tempdir().unwrap();
        let lib = dir.path().join("library");
        std::fs::create_dir_all(lib.join("a")).unwrap();
        std::fs::create_dir_all(lib.join("b")).unwrap();
        let display = lib.join("a/DSCF0001.JPG");
        let raw = lib.join("b/DSCF0001.RAF");
        std::fs::write(&display, b"display").unwrap();
        std::fs::write(&raw, b"negative").unwrap();
        let db = Database::open(&dir.path().join("library.db")).unwrap();
        db.insert_image_with_raw(&display, &[0x11u8; 32], 7, Some(&raw))
            .unwrap();

        let report = db.repair_split_companions().unwrap();
        assert_eq!(report, RepairReport { moved: 1, ghosts_deleted: 0, blocked: 0 });
        assert!(lib.join("a/DSCF0001.RAF").exists());
    }
}
