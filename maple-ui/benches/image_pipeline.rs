//! End-to-end image pipeline benchmarks.
//!
//! Measures each stage from bytes on disk to a displayable Slint `Image` so
//! that regressions and speedups appear as isolated numbers.
//!
//! Stages (in pipeline order):
//!   1. `bytes_read`     — `loadable_image_bytes` (file I/O only, no decode)
//!   2. `jpeg_decode`    — `decode_image_bytes` on in-memory JPEG bytes
//!   3. `render_to_rgb`  — decode + Lanczos3 resize to 256 px (cache-miss path)
//!   4. `webp_encode_80` — lossy WebP encode of the 256-px RGB at quality 80
//!   5. `webp_encode_95` — same at quality 95 (spot-check quality vs speed)
//!   6. `webp_decode`    — decode a stored WebP thumbnail back to RGB
//!   7. `thumbnail_full` — full pipeline: decode + resize + WebP encode
//!
//! All stages operate on a synthetic 3000×2000 JPEG created at bench startup.
//! No fixture files need to be committed to the repository.
//!
//! Run a single stage:
//!   cargo bench -p maple-ui -- image_pipeline/jpeg_decode

use std::io::Cursor;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use maple_import::{decode_image_bytes, loadable_image_bytes};
use maple_ui::thumbnail::{decode_webp_rgb, encode_webp_rgb, generate_thumbnail, render_to_rgb};

// ── Fixture ───────────────────────────────────────────────────────────────────

const BENCH_W: u32 = 3000;
const BENCH_H: u32 = 2000;

struct Fixture {
    _dir: tempfile::TempDir,
    path: PathBuf,
    jpeg_bytes: Vec<u8>,
}

fn make_fixture() -> Fixture {
    let img = image::RgbImage::from_fn(BENCH_W, BENCH_H, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    let mut buf = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut buf, image::ImageFormat::Jpeg)
        .expect("fixture encode failed");
    let jpeg_bytes = buf.into_inner();

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bench.jpg");
    std::fs::write(&path, &jpeg_bytes).expect("fixture write");

    Fixture { _dir: dir, path, jpeg_bytes }
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_bytes_read(c: &mut Criterion) {
    let f = make_fixture();
    let mut g = c.benchmark_group("image_pipeline");
    g.throughput(Throughput::Bytes(f.jpeg_bytes.len() as u64));
    g.bench_function("bytes_read", |b| {
        b.iter(|| loadable_image_bytes(&f.path).expect("read"))
    });
    g.finish();
}

fn bench_jpeg_decode(c: &mut Criterion) {
    let f = make_fixture();
    let mut g = c.benchmark_group("image_pipeline");
    g.throughput(Throughput::Bytes(f.jpeg_bytes.len() as u64));
    g.bench_function("jpeg_decode", |b| {
        b.iter(|| decode_image_bytes(&f.jpeg_bytes).expect("decode"))
    });
    g.finish();
}

fn bench_render_to_rgb(c: &mut Criterion) {
    let f = make_fixture();
    let mut g = c.benchmark_group("image_pipeline");
    g.throughput(Throughput::Bytes(f.jpeg_bytes.len() as u64));
    g.bench_function("render_to_rgb_256", |b| {
        b.iter(|| render_to_rgb(&f.path, 256).expect("render"))
    });
    g.finish();
}

fn bench_webp_encode(c: &mut Criterion) {
    let f = make_fixture();
    let (rgb, w, h) = render_to_rgb(&f.path, 256).expect("render for encode bench");

    let mut g = c.benchmark_group("image_pipeline");
    g.throughput(Throughput::Bytes(rgb.len() as u64));
    for quality in [80u8, 95] {
        g.bench_with_input(
            BenchmarkId::new("webp_encode", quality),
            &quality,
            |b, &q| b.iter(|| encode_webp_rgb(&rgb, w, h, q)),
        );
    }
    g.finish();
}

fn bench_webp_decode(c: &mut Criterion) {
    let f = make_fixture();
    let webp = generate_thumbnail(&f.path, 256, 80).expect("gen thumbnail for decode bench");

    let mut g = c.benchmark_group("image_pipeline");
    g.throughput(Throughput::Bytes(webp.len() as u64));
    g.bench_function("webp_decode", |b| {
        b.iter(|| decode_webp_rgb(&webp).expect("decode webp"))
    });
    g.finish();
}

fn bench_thumbnail_full(c: &mut Criterion) {
    let f = make_fixture();
    let mut g = c.benchmark_group("image_pipeline");
    g.throughput(Throughput::Bytes(f.jpeg_bytes.len() as u64));
    g.bench_function("thumbnail_full", |b| {
        b.iter(|| generate_thumbnail(&f.path, 256, 80).expect("generate_thumbnail"))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_bytes_read,
    bench_jpeg_decode,
    bench_render_to_rgb,
    bench_webp_encode,
    bench_webp_decode,
    bench_thumbnail_full,
);
criterion_main!(benches);
