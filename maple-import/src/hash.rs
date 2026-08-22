//! Content hashing for image deduplication.
//!
//! Uses BLAKE3 (streaming) and returns the full 32-byte digest for storage
//! in [`maple_state::SeenSet`].  A 32-byte hash makes collisions
//! cryptographically implausible (2⁻²⁵⁶ per pair).

use std::io::Read;
use std::path::Path;

/// Compute the full 32-byte BLAKE3 hash of the file at `path`.
///
/// Streams the file in 64 KiB chunks to avoid loading it entirely into
/// memory.
pub fn content_hash(path: &Path) -> anyhow::Result<[u8; 32]> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(*hasher.finalize().as_bytes())
}

/// Hash bytes already in memory.
///
/// The streaming reader above exists so a 60 MB raw file is not held in
/// memory just to be hashed. This one is for the case where the bytes are
/// already there and there is no file yet: a downloaded original, which sync
/// must verify *before* it writes it into a library. Both produce the same
/// digest for the same content — that is the whole point of using it as the
/// transfer's identifier.
pub fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_and_file_agree() {
        // Sync addresses blobs by the hash of the file on the sender and
        // checks the hash of the bytes on the receiver. If these two ever
        // disagreed, every transfer would be rejected as corrupt.
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("f.bin");
        let content = b"a photo, for the sake of argument";
        std::fs::write(&f, content).unwrap();

        assert_eq!(content_hash(&f).unwrap(), hash_bytes(content));
    }

    #[test]
    fn same_content_same_hash() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.bin");
        let b = dir.path().join("b.bin");
        std::fs::write(&a, b"identical content").unwrap();
        std::fs::write(&b, b"identical content").unwrap();

        assert_eq!(content_hash(&a).unwrap(), content_hash(&b).unwrap());
    }

    #[test]
    fn different_content_different_hash() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.bin");
        let b = dir.path().join("b.bin");
        std::fs::write(&a, b"content A").unwrap();
        std::fs::write(&b, b"content B").unwrap();

        assert_ne!(content_hash(&a).unwrap(), content_hash(&b).unwrap());
    }

    #[test]
    fn missing_file_returns_error() {
        assert!(content_hash(Path::new("/nonexistent/file")).is_err());
    }

    #[test]
    fn returns_32_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("f.bin");
        std::fs::write(&f, b"data").unwrap();
        let h = content_hash(&f).unwrap();
        assert_eq!(h.len(), 32);
    }
}
