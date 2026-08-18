//! Measurement harness for the library listing query.
//!
//! `#[ignore]`d: seeding a realistically sized library takes tens of seconds
//! and the timings are only meaningful in a release build.  Run with
//!
//! ```sh
//! cargo test --release -p maple-db -- --ignored --nocapture listing_bench
//! ```

use crate::{Database, SearchOrder, SearchQuery};
use rusqlite::Connection;
use std::time::Instant;

// ── Seeding ──────────────────────────────────────────────────────

/// Deterministic LCG — the seeded library must be identical run to run or
/// before/after numbers are not comparable.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Insert `n` image rows with a plausible mix: 5% missing, 20% with no
/// capture date, 10% grouped into stacks of four (half of which carry an
/// explicit cover), and `added_at` values that tie in threes the way a bulk
/// import produces them.
fn seed(conn: &Connection, n: usize) {
    let mut rng = Rng(0x5eed);
    let tx = conn.unchecked_transaction().expect("tx");

    let n_stacks = n / 40;
    {
        let mut ins_stack = tx
            .prepare("INSERT INTO stacks(id, created_at) VALUES (?1, ?2)")
            .expect("prepare stacks");
        for s in 1..=n_stacks {
            ins_stack.execute(rusqlite::params![s as i64, 1_700_000_000i64]).expect("stack");
        }
    }

    {
        let mut ins = tx
            .prepare(
                "INSERT INTO images
                     (id, path, hash, file_size, added_at, status, filename,
                      taken_at, make, model, lens, iso, width, height, stack_id)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
            )
            .expect("prepare images");

        let makes = ["Canon", "Nikon", "Fujifilm", "Sony", "Panasonic"];
        let models = ["EOS R5", "Z9", "X-T5", "A7 IV", "GH6"];
        let lenses = ["24-70mm f/2.8", "50mm f/1.8", "16-55mm f/2.8", "85mm f/1.4"];

        for i in 0..n {
            let id = (i + 1) as i64;
            // Ties in threes, and a slow march backwards in time.
            let added_at = 1_700_000_000i64 - (i as i64 / 3) * 11;
            let taken_at = if rng.below(100) < 20 {
                None
            } else {
                // Shuffled relative to added_at so the two orderings differ.
                Some(added_at - rng.below(90 * 86_400) as i64)
            };
            let status = if rng.below(100) < 5 { "missing" } else { "present" };
            // The first 10% of ids are the stack members: four per stack.
            let stack_id = if i < n_stacks * 4 { Some((i / 4 + 1) as i64) } else { None };
            let k = i % makes.len();

            ins.execute(rusqlite::params![
                id,
                format!("/photos/{:04}/IMG_{id:07}.jpg", i / 500),
                vec![(i % 251) as u8; 32],
                4_000_000i64 + i as i64,
                added_at,
                status,
                format!("IMG_{id:07}.jpg"),
                taken_at,
                makes[k],
                models[k],
                lenses[i % lenses.len()],
                100i64 * (1 + (i % 8) as i64),
                6000i64,
                4000i64,
                stack_id,
            ])
            .expect("image");
        }
    }

    // Half the stacks pin an explicit cover (a member other than the lowest id).
    tx.execute(
        "UPDATE stacks SET cover_image_id = id * 4 WHERE id % 2 = 0 AND id * 4 <= ?1",
        rusqlite::params![(n_stacks * 4) as i64],
    )
    .expect("covers");

    // A collection holding ~2% of the library, and a person on ~1%.
    tx.execute(
        "INSERT INTO collections(id, name, color, created_at) VALUES (1, 'Bench', '#fff', 0)",
        [],
    )
    .expect("collection");
    tx.execute(
        "INSERT INTO collection_images(collection_id, image_id, added_at)
         SELECT 1, id, 0 FROM images WHERE id % 50 = 0",
        [],
    )
    .expect("collection members");
    tx.execute(
        "INSERT INTO persons(id, name, created_at) VALUES (1, 'Bench Person', 0)",
        [],
    )
    .expect("person");
    tx.execute(
        "INSERT INTO face_detections
             (image_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding, person_id, confidence)
         SELECT id, 0.1, 0.1, 0.4, 0.4, X'00', 1, 0.9 FROM images WHERE id % 100 = 0",
        [],
    )
    .expect("faces");

    tx.commit().expect("commit");
}

fn seeded_db(n: usize) -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(&dir.path().join("library.db")).expect("open");
    let t = Instant::now();
    seed(&db.conn, n);
    eprintln!("seeded {n} rows in {:?}", t.elapsed());
    (dir, db)
}

// ── Timing helpers ───────────────────────────────────────────────

/// Best of five runs, in milliseconds — the listing runs against a warm page
/// cache in the app too (the grid re-issues it constantly while scrolling).
fn best_ms(mut f: impl FnMut() -> usize) -> (f64, usize) {
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..5 {
        let t = Instant::now();
        rows = f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    (best, rows)
}

fn explain(conn: &Connection, sql: &str, params: &[rusqlite::types::Value]) -> String {
    let mut stmt = match conn.prepare(&format!("EXPLAIN QUERY PLAN {sql}")) {
        Ok(s) => s,
        Err(e) => return format!("<prepare failed: {e}>"),
    };
    let rows: Vec<String> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |r| r.get::<_, String>(3))
        .expect("eqp")
        .filter_map(|r| r.ok())
        .collect();
    rows.join("\n    ")
}

// ── The measurements ─────────────────────────────────────────────

const N: usize = 200_000;

#[test]
#[ignore]
fn listing_bench() {
    let (_dir, db) = seeded_db(N);

    for order in [SearchOrder::AddedDesc, SearchOrder::TakenDesc] {
        eprintln!("\n=== {order:?} ===");
        for offset in [0usize, 15_000, 150_000] {
            let q = SearchQuery::default().with_limit(300).with_offset(offset).with_order(order);
            let (ms, rows) = best_ms(|| db.search_images(&q).expect("search").len());
            eprintln!("  offset {offset:>7}: {ms:8.2} ms ({rows} rows)");
        }
    }

    let q = SearchQuery::default().with_limit(300).with_order(SearchOrder::AddedDesc);
    let (ms, _) = best_ms(|| db.count_images(&q).expect("count").unwrap_or(0));
    eprintln!("\ncount_images: {ms:8.2} ms");

    for order in [SearchOrder::AddedDesc, SearchOrder::TakenDesc] {
        let (from_where, mut params) = crate::all_from_where(crate::Entry::Index, None, None);
        let order_by = crate::query::order_by_sql(order);
        let sql =
            format!("SELECT {} {from_where} {order_by} LIMIT ? OFFSET ?", crate::IMAGE_COLUMNS);
        params.push(rusqlite::types::Value::Integer(300));
        params.push(rusqlite::types::Value::Integer(15_000));
        eprintln!("\n{order:?} plan:\n    {}", explain(&db.conn, &sql, &params));
    }
}

// ── Restructuring experiment ─────────────────────────────────────

const IDX_ADDED: &str = "CREATE INDEX IF NOT EXISTS idx_images_listing_added \
     ON images(status, added_at DESC, id DESC)";
const IDX_TAKEN: &str = "CREATE INDEX IF NOT EXISTS idx_images_listing_taken \
     ON images(status, COALESCE(taken_at, added_at) DESC, id DESC)";

/// The pre-V17 listing: stack covers as a materialised CTE, joined in, with
/// the cover test spanning the join.  Kept verbatim so "before" stays
/// measurable after the production query moved on.
fn sql_cte(order_by: &str) -> String {
    format!(
        "WITH stack_covers AS (
             SELECT s.id                                   AS stack_id,
                    COALESCE(s.cover_image_id, MIN(si.id)) AS cover_id,
                    COUNT(*)                               AS stack_size
             FROM stacks s
             JOIN images si ON si.stack_id = s.id AND si.status = 'present'
             GROUP BY s.id
         )
         SELECT i.id, i.path, i.added_at, i.status,
                i.filename, i.taken_at, i.make, i.model, i.lens,
                i.focal_length, i.aperture, i.iso,
                i.width, i.height, i.orientation, i.raw_path, i.hash,
                i.stack_id, sc.stack_size
         FROM images i
         LEFT JOIN stack_covers sc ON sc.stack_id = i.stack_id
         WHERE i.status = 'present'
           AND (i.stack_id IS NULL OR i.id = sc.cover_id)
         {order_by} LIMIT ? OFFSET ?"
    )
}

/// The shipped listing: cover eligibility and stack size as correlated
/// subqueries over `i` alone, no join to reach across.
fn sql_correlated(order_by: &str) -> String {
    let (from_where, _) = crate::all_from_where(crate::Entry::Index, None, None);
    format!("SELECT {} {from_where} {order_by} LIMIT ? OFFSET ?", crate::IMAGE_COLUMNS)
}

/// Remove the V17 indexes so the "before" stage measures the pre-V17 planner.
fn drop_listing_indexes(conn: &Connection) {
    conn.execute_batch(
        "DROP INDEX IF EXISTS idx_images_listing_added;
         DROP INDEX IF EXISTS idx_images_listing_taken;",
    )
    .expect("drop indexes");
}

fn run(conn: &Connection, sql: &str, limit: i64, offset: i64) -> Vec<i64> {
    let mut stmt = conn.prepare(sql).expect("prepare");
    stmt.query_map(rusqlite::params![limit, offset], |r| r.get::<_, i64>(0))
        .expect("query")
        .filter_map(|r| r.ok())
        .collect()
}

#[test]
#[ignore]
fn index_experiment() {
    let (_dir, db) = seeded_db(N);
    let conn = &db.conn;
    drop_listing_indexes(conn);

    let orders = [
        ("AddedDesc", "ORDER BY i.added_at DESC, i.id DESC"),
        ("TakenDesc", "ORDER BY COALESCE(i.taken_at, i.added_at) DESC, i.id DESC"),
    ];

    for stage in ["no indexes", "with indexes"] {
        if stage == "with indexes" {
            let t = Instant::now();
            conn.execute_batch(IDX_ADDED).expect("idx added");
            conn.execute_batch(IDX_TAKEN).expect("idx taken");
            eprintln!("\n### index build over {N} rows: {:?}", t.elapsed());
        }
        eprintln!("\n########## {stage} ##########");

        for (name, order_by) in orders {
            for (variant, sql) in
                [("CTE+join ", sql_cte(order_by)), ("correlated", sql_correlated(order_by))]
            {
                let plan = explain(
                    conn,
                    &sql,
                    &[rusqlite::types::Value::Integer(300), rusqlite::types::Value::Integer(15_000)],
                );
                let temp_btree = plan.contains("TEMP B-TREE FOR ORDER BY");
                eprint!("\n{name} / {variant}  temp-b-tree={temp_btree}\n    {plan}\n   ");
                for offset in [0i64, 15_000, 150_000] {
                    let (ms, _) = best_ms(|| run(conn, &sql, 300, offset).len());
                    eprint!("  off {offset}: {ms:7.2} ms");
                }
                eprintln!();
            }
        }
    }

    // Equivalence at scale: the two shapes must return the same ids in the
    // same order, in both sort modes, filtered and not — and consecutive
    // pages must tile the listing with no gap or repeat across the boundary.
    let filters = [
        ("unfiltered", ""),
        (
            "collection",
            " AND i.id IN (SELECT image_id FROM collection_images WHERE collection_id = 1)",
        ),
        ("person", " AND i.id IN (SELECT image_id FROM face_detections WHERE person_id = 1)"),
    ];
    for (label, extra) in filters {
        for (order, order_by) in orders {
            let splice = |sql: String| sql.replacen("ORDER BY", &format!("{extra} ORDER BY"), 1);
            let before = splice(sql_cte(order_by));
            let after = splice(sql_correlated(order_by));

            for offset in [0i64, 15_000, 150_000] {
                assert_eq!(
                    run(conn, &before, 300, offset),
                    run(conn, &after, 300, offset),
                    "{order}/{label} disagrees at offset {offset}"
                );
            }

            // Three consecutive pages, concatenated, equal one 900-row read.
            let whole = run(conn, &after, 900, 0);
            let paged: Vec<i64> =
                (0..3).flat_map(|p| run(conn, &after, 300, p * 300)).collect();
            assert_eq!(paged, whole, "{order}/{label} pages do not tile the listing");
        }
    }
    eprintln!("\nequivalence: OK");
}

/// Does the index have to spell out DESC, and does a partial index on
/// `status = 'present'` do the job more cheaply?
#[test]
#[ignore]
fn index_shape_experiment() {
    let shapes: [(&str, &[&str]); 3] = [
        (
            "composite DESC",
            &[
                "CREATE INDEX ix_a ON images(status, added_at DESC, id DESC)",
                "CREATE INDEX ix_t ON images(status, COALESCE(taken_at, added_at) DESC, id DESC)",
            ],
        ),
        (
            "composite ASC",
            &[
                "CREATE INDEX ix_a ON images(status, added_at, id)",
                "CREATE INDEX ix_t ON images(status, COALESCE(taken_at, added_at), id)",
            ],
        ),
        (
            "partial (status='present')",
            &[
                "CREATE INDEX ix_a ON images(added_at DESC, id DESC) WHERE status = 'present'",
                "CREATE INDEX ix_t ON images(COALESCE(taken_at, added_at) DESC, id DESC) \
                 WHERE status = 'present'",
            ],
        ),
    ];

    for (name, ddl) in shapes {
        let (_dir, db) = seeded_db(N);
        let conn = &db.conn;
        drop_listing_indexes(conn);
        let t = Instant::now();
        for s in ddl {
            conn.execute_batch(s).expect("index");
        }
        let build = t.elapsed();
        let size: i64 = conn
            .query_row(
                "SELECT SUM(pgsize) FROM dbstat WHERE name IN ('ix_a','ix_t')",
                [],
                |r| r.get(0),
            )
            .unwrap_or(-1);
        eprintln!("\n--- {name} (build {build:?}, {size} bytes) ---");

        for (order, order_by) in [
            ("AddedDesc", "ORDER BY i.added_at DESC, i.id DESC"),
            ("TakenDesc", "ORDER BY COALESCE(i.taken_at, i.added_at) DESC, i.id DESC"),
        ] {
            let sql = sql_correlated(order_by);
            let plan = explain(
                conn,
                &sql,
                &[rusqlite::types::Value::Integer(300), rusqlite::types::Value::Integer(0)],
            );
            let (ms, _) = best_ms(|| run(conn, &sql, 300, 0).len());
            eprintln!(
                "  {order}: off 0 {ms:6.2} ms  temp-b-tree={}  [{}]",
                plan.contains("TEMP B-TREE FOR ORDER BY"),
                plan.lines().next().unwrap_or("").trim()
            );
        }
    }
}

/// Collection- and person-filtered listings, and `count_images`, against the
/// restructured query — the filters are `IN (SELECT …)` subqueries applied on
/// top of an index-ordered scan, so they are worth a separate look.
#[test]
#[ignore]
fn filtered_listing_experiment() {
    let (_dir, db) = seeded_db(N);
    let conn = &db.conn;

    let filters = [
        ("unfiltered", String::new()),
        (
            "collection (2%)",
            " AND i.id IN (SELECT image_id FROM collection_images WHERE collection_id = 1)"
                .to_owned(),
        ),
        (
            "person (1%)",
            " AND i.id IN (SELECT image_id FROM face_detections WHERE person_id = 1)".to_owned(),
        ),
    ];

    drop_listing_indexes(conn);
    for stage in ["no indexes", "with indexes"] {
        if stage == "with indexes" {
            conn.execute_batch(IDX_ADDED).expect("idx");
            conn.execute_batch(IDX_TAKEN).expect("idx");
        }
        eprintln!("\n########## {stage} ##########");
        for (fname, extra) in &filters {
            for (variant, base) in [
                ("CTE+join ", sql_cte("ORDER BY i.added_at DESC, i.id DESC")),
                ("correlated", sql_correlated("ORDER BY i.added_at DESC, i.id DESC")),
            ] {
                // Splice the filter in ahead of the ORDER BY.
                let sql = base.replacen("ORDER BY", &format!("{extra} ORDER BY"), 1);
                assert!(extra.is_empty() || sql != base, "filter not spliced");
                let (ms, rows) = best_ms(|| run(conn, &sql, 300, 0).len());
                eprintln!("  {fname:>16} / {variant}: {ms:7.2} ms ({rows} rows)");
            }
        }
    }
}

/// The correlated cover test costs one or two index probes per stacked row.
/// The paged listing only ever sees `limit + offset` rows, but `count_images`
/// has no `LIMIT` and pays it across the whole library — so compare shapes
/// for the count as well as the listing.
///
/// Result: the shipped pair wins both halves.  The CTE forms lose on the
/// listing because of the temp b-tree, and lose on the count too once the
/// V17 indexes exist (the CTE's own scan goes through them).
#[test]
#[ignore]
fn count_experiment() {
    let (_dir, db) = seeded_db(N);
    let conn = &db.conn;

    // Cover eligibility against a materialised CTE, but as a row-value `IN`
    // over `i` alone rather than a join the ORDER BY has to reach across.
    const CTE_IN_FROM_WHERE: &str = "FROM images i
         WHERE i.status = 'present'
           AND (i.stack_id IS NULL
                OR (i.stack_id, i.id) IN (SELECT stack_id, cover_id FROM stack_covers))";
    const CTE_IN: &str = "WITH stack_covers AS (
             SELECT s.id                                   AS stack_id,
                    COALESCE(s.cover_image_id, MIN(si.id)) AS cover_id
             FROM stacks s
             JOIN images si ON si.stack_id = s.id AND si.status = 'present'
             GROUP BY s.id
         )";

    let (shipped, _) = crate::all_from_where(crate::Entry::Index, None, None);
    let order_by = crate::query::order_by_sql(SearchOrder::AddedDesc);
    let cols = crate::IMAGE_COLUMNS;

    const CTE_JOIN: &str = "WITH stack_covers AS (
             SELECT s.id                                   AS stack_id,
                    COALESCE(s.cover_image_id, MIN(si.id)) AS cover_id,
                    COUNT(*)                               AS stack_size
             FROM stacks s
             JOIN images si ON si.stack_id = s.id AND si.status = 'present'
             GROUP BY s.id
         )";
    const CTE_JOIN_FROM_WHERE: &str = "FROM images i
         LEFT JOIN stack_covers sc ON sc.stack_id = i.stack_id
         WHERE i.status = 'present'
           AND (i.stack_id IS NULL OR i.id = sc.cover_id)";

    let shapes = [
        (
            "correlated, index entry",
            format!("SELECT COUNT(*) {shipped}"),
            format!("SELECT {cols} {shipped} {order_by} LIMIT 300 OFFSET 15000"),
        ),
        (
            "CTE + row-value IN",
            format!("{CTE_IN} SELECT COUNT(*) {CTE_IN_FROM_WHERE}"),
            format!("{CTE_IN} SELECT {cols} {CTE_IN_FROM_WHERE} {order_by} LIMIT 300 OFFSET 15000"),
        ),
        (
            "correlated, NOT INDEXED (shipped count)",
            format!("SELECT COUNT(*) {}", shipped.replacen("FROM images i", "FROM images i NOT INDEXED", 1)),
            format!("SELECT {cols} {shipped} {order_by} LIMIT 300 OFFSET 15000"),
        ),
        (
            "correlated, split sum",
            format!(
                "SELECT (SELECT COUNT(*) FROM images WHERE status = 'present' AND stack_id IS NULL)
                      + (SELECT COUNT(*) FROM images i
                         WHERE i.status = 'present' AND i.stack_id IS NOT NULL
                           AND {})",
                crate::STACK_COVER_PREDICATE
            ),
            format!("SELECT {cols} {shipped} {order_by} LIMIT 300 OFFSET 15000"),
        ),
        (
            "CTE + join (pre-V17)",
            format!("{CTE_JOIN} SELECT COUNT(*) {CTE_JOIN_FROM_WHERE}"),
            format!(
                "{CTE_JOIN} SELECT {cols} {CTE_JOIN_FROM_WHERE} {order_by} LIMIT 300 OFFSET 15000"
            ),
        ),
    ];

    for (name, count_sql, list_sql) in &shapes {
        let n: i64 = conn.query_row(count_sql, [], |r| r.get(0)).expect("count");
        let (cms, _) = best_ms(|| {
            conn.query_row(count_sql, [], |r| r.get::<_, i64>(0)).expect("count") as usize
        });
        let plan = explain(conn, list_sql, &[]);
        let count_plan = explain(conn, count_sql, &[]);
        let (lms, rows) = best_ms(|| {
            let mut stmt = conn.prepare(list_sql).expect("prepare");
            stmt.query_map([], |r| r.get::<_, i64>(0)).expect("q").filter_map(|r| r.ok()).count()
        });
        eprintln!(
            "\n{name}\n  count: {cms:7.2} ms (n = {n})\n  list : {lms:7.2} ms ({rows} rows)  \
             temp-b-tree={}\n    {plan}\n  count plan:\n    {count_plan}",
            plan.contains("TEMP B-TREE FOR ORDER BY")
        );
    }
}

/// `count_images` has no `LIMIT`, so it needs `stack_id` for every present
/// row.  The V17 indexes do not carry it, so the planner scans one of them
/// and then fetches each row from the table in index order — random reads
/// across the whole file.  Two ways out: forbid the index (`NOT INDEXED`) so
/// it scans the table sequentially, or widen the index to carry `stack_id`.
///
/// Widening is a trap, which is why V17 does not do it and `Entry::Table`
/// exists instead: a widened index is *covering*, and the planner then
/// prefers it over `idx_images_stack_id` for the per-row stack-size
/// subquery — page 0 of the listing went from 0.8 ms to 1300 ms.  Widening
/// only ever looks good if you measure the count in isolation.
#[test]
#[ignore]
fn count_index_width_experiment() {
    for (name, ddl) in [
        (
            "V17 as shipped",
            "CREATE INDEX IF NOT EXISTS idx_images_listing_added \
                 ON images(status, added_at DESC, id DESC)",
        ),
        (
            "widened with stack_id",
            "CREATE INDEX IF NOT EXISTS idx_images_listing_added \
                 ON images(status, added_at DESC, id DESC, stack_id)",
        ),
    ] {
        let (_dir, db) = seeded_db(N);
        let conn = &db.conn;
        drop_listing_indexes(conn);
        conn.execute_batch(ddl).expect("idx");
        conn.execute_batch(IDX_TAKEN).expect("idx taken");

        let (shipped, _) = crate::all_from_where(crate::Entry::Index, None, None);
        let plain = format!("SELECT COUNT(*) {shipped}");
        let hinted = plain.replacen("FROM images i", "FROM images i NOT INDEXED", 1);

        eprintln!("\n--- {name} ---");
        for (label, sql) in [("plain", &plain), ("NOT INDEXED", &hinted)] {
            let n: i64 = conn.query_row(sql, [], |r| r.get(0)).expect("count");
            let (ms, _) = best_ms(|| {
                conn.query_row(sql, [], |r| r.get::<_, i64>(0)).expect("count") as usize
            });
            eprintln!("  unfiltered {label:>12}: {ms:7.2} ms (n = {n})");
        }

        // Filtered counts must not be collateral damage of the hint.
        for (fname, cid, pid) in [("collection", Some(1i64), None), ("person", None, Some(1i64))] {
            let (fw, params) = crate::all_from_where(crate::Entry::Index, cid, pid);
            let plain = format!("SELECT COUNT(*) {fw}");
            let hinted = plain.replacen("FROM images i", "FROM images i NOT INDEXED", 1);
            for (label, sql) in [("plain", &plain), ("NOT INDEXED", &hinted)] {
                let n: i64 = conn
                    .query_row(sql, rusqlite::params_from_iter(params.iter()), |r| r.get(0))
                    .expect("count");
                let (ms, _) = best_ms(|| {
                    conn.query_row(sql, rusqlite::params_from_iter(params.iter()), |r| {
                        r.get::<_, i64>(0)
                    })
                    .expect("count") as usize
                });
                eprintln!("  {fname:>10} {label:>12}: {ms:7.2} ms (n = {n})");
            }
        }
    }
}

/// Keyset ("seek") paging against the same indexes: instead of discarding
/// `offset` rows, resume from the last row of the previous page.
#[test]
#[ignore]
fn keyset_experiment() {
    let (_dir, db) = seeded_db(N);
    let conn = &db.conn;

    let cover_pred = "(i.stack_id IS NULL
                OR i.id = (SELECT COALESCE(s.cover_image_id,
                                  (SELECT MIN(m.id) FROM images m
                                   WHERE m.stack_id = i.stack_id AND m.status = 'present'))
                           FROM stacks s WHERE s.id = i.stack_id))";

    // Same column list the real listing materialises, so the comparison
    // against OFFSET measures only the paging strategy.
    let cols = format!(
        "i.id, {}, CASE WHEN i.stack_id IS NOT NULL THEN
             (SELECT COUNT(*) FROM images m
              WHERE m.stack_id = i.stack_id AND m.status = 'present')
         ELSE NULL END",
        "i.path, i.added_at, i.status, i.filename, i.taken_at, i.make, i.model, \
         i.lens, i.focal_length, i.aperture, i.iso, i.width, i.height, \
         i.orientation, i.raw_path, i.hash, i.stack_id"
    );

    for (name, key, order_by) in [
        ("AddedDesc", "i.added_at", "ORDER BY i.added_at DESC, i.id DESC"),
        (
            "TakenDesc",
            "COALESCE(i.taken_at, i.added_at)",
            "ORDER BY COALESCE(i.taken_at, i.added_at) DESC, i.id DESC",
        ),
    ] {
        let seek = format!(
            "SELECT {cols}, {key} FROM images i
             WHERE i.status = 'present' AND {cover_pred}
               AND ({key} < ?1 OR ({key} = ?1 AND i.id < ?2))
             {order_by} LIMIT ?3"
        );
        let first = format!(
            "SELECT {cols}, {key} FROM images i
             WHERE i.status = 'present' AND {cover_pred}
             {order_by} LIMIT ?1"
        );

        eprintln!("\n{name} seek plan:\n    {}", explain(conn, &seek, &[
            rusqlite::types::Value::Integer(0),
            rusqlite::types::Value::Integer(0),
            rusqlite::types::Value::Integer(300),
        ]));

        // Walk 500 pages and time the last one — the case OFFSET degrades on.
        let mut stmt_first = conn.prepare(&first).expect("prepare first");
        let mut cursor: Option<(i64, i64)> = stmt_first
            .query_map(rusqlite::params![300i64], |r| {
                Ok((r.get::<_, i64>(19)?, r.get::<_, i64>(0)?))
            })
            .expect("first")
            .filter_map(|r| r.ok())
            .last();

        let mut page = 0;
        let mut deep_ms = 0.0f64;
        while let Some((k, id)) = cursor {
            page += 1;
            let t = Instant::now();
            let mut stmt = conn.prepare(&seek).expect("prepare seek");
            let rows: Vec<(i64, i64)> = stmt
                .query_map(rusqlite::params![k, id, 300i64], |r| {
                    Ok((r.get::<_, i64>(19)?, r.get::<_, i64>(0)?))
                })
                .expect("seek")
                .filter_map(|r| r.ok())
                .collect();
            deep_ms = t.elapsed().as_secs_f64() * 1000.0;
            if rows.len() < 300 || page >= 500 {
                eprintln!("  page {page}: {deep_ms:7.2} ms (last page, {} rows)", rows.len());
                break;
            }
            cursor = rows.last().copied();
        }
        eprintln!("  {name}: {page} pages walked, last page {deep_ms:.2} ms");
    }
}

/// Text search: LIKE '%token%' across four expressions and three LEFT JOINs.
#[test]
#[ignore]
fn text_search_bench() {
    let (_dir, db) = seeded_db(N);

    for text in ["fujifilm", "fujifilm 85mm", "IMG_0100000"] {
        let q = SearchQuery::default().with_text(text).with_limit(300);
        let (ms, rows) = best_ms(|| db.search_images(&q).expect("search").len());
        eprintln!("LIKE  {text:>16}: {ms:8.2} ms ({rows} rows)");
    }

    // The same tokens through an FTS5 index over the same EXIF columns, to
    // price what V17 gave up by dropping `image_fts` (see schema.rs).
    build_fts_index(&db.conn);
    for text in ["fujifilm", "fujifilm 85mm", "IMG_0100000"] {
        let fts = fts_match_expr(text);
        let (ms, rows) = best_ms(|| {
            let mut stmt = db
                .conn
                .prepare(
                    "SELECT rowid FROM image_fts WHERE image_fts MATCH ?1 LIMIT 300",
                )
                .expect("prepare fts");
            stmt.query_map(rusqlite::params![fts], |r| r.get::<_, i64>(0))
                .expect("fts")
                .filter_map(|r| r.ok())
                .count()
        });
        eprintln!("FTS5  {text:>16}: {ms:8.2} ms ({rows} rows)");
    }
}

/// Insert throughput with the FTS5 sync triggers in place versus without —
/// the write-side cost of a table no read path queries.
/// Rebuild the V2-era `image_fts` index (dropped by V17) over the seeded rows.
fn build_fts_index(conn: &Connection) {
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS image_fts USING fts5(
             filename, make, model, lens, tokenize='unicode61');
         INSERT INTO image_fts(rowid, filename, make, model, lens)
             SELECT id, filename, make, model, lens FROM images;",
    )
    .expect("fts index");
}

/// Each token as an FTS5 prefix match — note this is *prefix*, where the
/// production `LIKE '%token%'` path matches mid-word too.
fn fts_match_expr(text: &str) -> String {
    text.split_whitespace()
        .map(|t| format!("\"{}\"*", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore]
fn fts_write_cost_bench() {
    let insert_batch = |with_triggers: bool| -> f64 {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(&dir.path().join("library.db")).expect("open");
        if with_triggers {
            db.conn
                .execute_batch(
                    "CREATE VIRTUAL TABLE image_fts USING fts5(
                         filename, make, model, lens, tokenize='unicode61');
                     CREATE TRIGGER images_fts_ai AFTER INSERT ON images BEGIN
                         INSERT INTO image_fts(rowid, filename, make, model, lens)
                         VALUES (new.id, new.filename, new.make, new.model, new.lens);
                     END;
                     CREATE TRIGGER images_fts_au AFTER UPDATE ON images BEGIN
                         DELETE FROM image_fts WHERE rowid = old.id;
                         INSERT INTO image_fts(rowid, filename, make, model, lens)
                         VALUES (new.id, new.filename, new.make, new.model, new.lens);
                     END;
                     CREATE TRIGGER images_fts_ad AFTER DELETE ON images BEGIN
                         DELETE FROM image_fts WHERE rowid = old.id;
                     END;",
                )
                .expect("fts triggers");
        }
        let t = Instant::now();
        seed(&db.conn, 100_000);
        t.elapsed().as_secs_f64() * 1000.0
    };

    let with = insert_batch(true);
    let without = insert_batch(false);
    eprintln!("100k inserts with FTS triggers:    {with:9.1} ms");
    eprintln!("100k inserts without FTS triggers: {without:9.1} ms");
    eprintln!("FTS overhead: {:.1}%", (with / without - 1.0) * 100.0);
}
