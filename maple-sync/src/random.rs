//! Where random bytes come from.
//!
//! The workspace has no `rand` dependency and doesn't want one: guids and
//! device ids are already minted from SQLite's `randomblob`, which is seeded
//! from the OS CSPRNG. Rather than reach for a second source, pairing takes
//! its randomness through this trait so the production wiring can hand it
//! `maple_db::Database::random_bytes` and tests can hand it something
//! reproducible.
//!
//! Injecting it is not only about testability. A pairing test that sampled a
//! real RNG could only assert "it didn't crash"; with the stream fixed, a
//! test can assert the *exact* bytes a handshake produces, which is how the
//! sealing step is checked to actually seal.

/// A source of cryptographically random bytes.
///
/// Implementations must be unpredictable to an attacker — the pairing secret,
/// the long-term key and every request nonce come from here.
pub trait RandomSource {
    /// Fill `buf` completely with random bytes.
    fn fill(&self, buf: &mut [u8]) -> anyhow::Result<()>;

    /// Convenience for the fixed-size cases (nonces, keys).
    ///
    /// `Sized` so the trait stays object-safe despite the const generic —
    /// see [`SharedRandom`], which needs `dyn RandomSource` to exist.
    fn array<const N: usize>(&self) -> anyhow::Result<[u8; N]>
    where
        Self: Sized,
    {
        let mut out = [0u8; N];
        self.fill(&mut out)?;
        Ok(out)
    }
}

/// A random source shared across threads.
///
/// The sync server hands its listener thread the same SQLite-backed source
/// the UI uses, and a generic parameter would have to be threaded through
/// every type that touches it. Erasing it here keeps `SyncServer` a plain
/// struct.
pub type SharedRandom = std::sync::Arc<dyn RandomSource + Send + Sync>;

/// So a [`SharedRandom`] can be passed wherever an `impl RandomSource` is
/// wanted — including [`array`](RandomSource::array), which the erased form
/// cannot offer on its own.
impl RandomSource for SharedRandom {
    fn fill(&self, buf: &mut [u8]) -> anyhow::Result<()> {
        (**self).fill(buf)
    }
}

/// Adapts a closure into a [`RandomSource`].
///
/// Lets a caller bridge to whatever it already has without either crate
/// depending on the other:
///
/// ```ignore
/// let rng = FnRandom(|buf: &mut [u8]| {
///     buf.copy_from_slice(&db.random_bytes(buf.len())?);
///     Ok(())
/// });
/// ```
pub struct FnRandom<F>(pub F);

impl<F> RandomSource for FnRandom<F>
where
    F: Fn(&mut [u8]) -> anyhow::Result<()>,
{
    fn fill(&self, buf: &mut [u8]) -> anyhow::Result<()> {
        (self.0)(buf)
    }
}

/// A reproducible byte stream for tests.
///
/// Deliberately **not** public: a seeded generator in a security crate is a
/// footgun in production, and the one legitimate use — asserting that a
/// handshake produces exactly the bytes it should — is internal.
///
/// Built on BLAKE3's XOF rather than a hand-rolled LCG so the stream is long,
/// well-distributed, and needs no extra dependency.
#[cfg(test)]
pub(crate) struct SeededRandom {
    reader: std::cell::RefCell<blake3::OutputReader>,
}

#[cfg(test)]
impl SeededRandom {
    pub(crate) fn new(seed: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"maple-sync-test-rng");
        hasher.update(&seed.to_le_bytes());
        Self {
            reader: std::cell::RefCell::new(hasher.finalize_xof()),
        }
    }
}

#[cfg(test)]
impl RandomSource for SeededRandom {
    fn fill(&self, buf: &mut [u8]) -> anyhow::Result<()> {
        self.reader.borrow_mut().fill(buf);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_streams_are_reproducible_and_seed_dependent() {
        let a: [u8; 32] = SeededRandom::new(7).array().unwrap();
        let b: [u8; 32] = SeededRandom::new(7).array().unwrap();
        let c: [u8; 32] = SeededRandom::new(8).array().unwrap();
        assert_eq!(a, b, "same seed must replay the same stream");
        assert_ne!(a, c);
    }

    #[test]
    fn seeded_stream_does_not_repeat_itself() {
        let rng = SeededRandom::new(1);
        let first: [u8; 16] = rng.array().unwrap();
        let second: [u8; 16] = rng.array().unwrap();
        assert_ne!(first, second, "successive draws must advance the stream");
    }

    #[test]
    fn fn_random_bridges_a_closure() {
        let rng = FnRandom(|buf: &mut [u8]| {
            buf.fill(0xAB);
            Ok(())
        });
        assert_eq!(rng.array::<4>().unwrap(), [0xAB; 4]);
    }
}
