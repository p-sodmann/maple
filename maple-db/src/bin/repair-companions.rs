//! Repair libraries where a synced companion raw was filed away from its
//! photo — and delete the duplicate rows the library scanner minted from it.
//!
//! ```sh
//! cargo run --release -p maple-db --bin repair-companions -- <library.db>
//! cargo run --release -p maple-db --bin repair-companions -- <library.db> --apply
//! ```
//!
//! Reports by default; `--apply` is what moves files and deletes rows. The
//! damage and the fix are described in `maple_db::repair`.
//!
//! **Run it on every device**, and let a sync pass run afterwards. The ghost
//! delete is tombstoned, so it replicates — but a peer that still holds its
//! *own* split pair has its own files to move, and no peer can do that for
//! it. Stop the app first: the 60-second library scanner is exactly the thing
//! that mints these rows, and one landing mid-repair would put a fresh ghost
//! behind the one just removed.

use std::path::PathBuf;

use maple_db::Database;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut args = std::env::args_os().skip(1);
    let Some(db_path) = args.next().map(PathBuf::from) else {
        eprintln!(
            "usage: repair-companions <library.db> [--apply]\n\n\
             Without --apply nothing is changed: the split pairs are listed and\n\
             the program exits."
        );
        std::process::exit(2);
    };
    let apply = args.any(|a| a == "--apply");

    let db = Database::open(&db_path)?;
    let split = db.split_companions()?;

    if split.is_empty() {
        println!("Nothing to repair: every companion sits beside its photo.");
        return Ok(());
    }

    println!("{} photo(s) whose companion is filed elsewhere:\n", split.len());
    for s in &split {
        println!("  image {}", s.image_id);
        println!("    display   {}", s.display.display());
        println!("    companion {}", s.raw.display());
        match &s.belongs_at {
            Some(home) => println!("    belongs   {}", home.display()),
            None => println!("    belongs   — that name is taken; needs a human"),
        }
        if !s.ghosts.is_empty() {
            // These are the rows that replicate the duplicate to every peer.
            println!("    duplicate row(s) the scanner minted: {:?}", s.ghosts);
        }
    }

    let blocked = split.iter().filter(|s| !s.actionable()).count();
    if !apply {
        println!(
            "\nReport only. Re-run with --apply to move {} file(s) and delete {} row(s).",
            split.len() - blocked,
            split.iter().map(|s| s.ghosts.len()).sum::<usize>(),
        );
        return Ok(());
    }

    let report = db.repair_split_companions()?;
    println!(
        "\nMoved {} companion(s), deleted {} duplicate row(s), left {} for a human.",
        report.moved, report.ghosts_deleted, report.blocked
    );
    if report.blocked > 0 {
        println!(
            "A blocked pair means the name beside the photo is already taken. \
             Rename or remove that file, then run this again."
        );
    }
    Ok(())
}
