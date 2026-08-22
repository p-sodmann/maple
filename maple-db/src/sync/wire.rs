//! Serialisable projections of the replicated tables.
//!
//! These types are the contract between two Maple installations, so they are
//! deliberately *not* mirrors of the SQL rows:
//!
//! * **Foreign keys travel as guids, never rowids.** An `INTEGER PRIMARY KEY`
//!   is assigned by whichever machine inserted the row, so `image_id = 42`
//!   means something different on every device.
//! * **Machine-local columns are absent.** `status`, `path`, `raw_path`,
//!   `filename`, `locality` and `origin_device` describe one machine's disk.
//!   A photo missing from the laptop is still present on the workstation, and
//!   replicating `'missing'` would let whichever device holds fewer originals
//!   blank out the other's library. `locality` is the same rule sharpened:
//!   the same photo is `local` on the master and `remote` on a relay servant,
//!   so shipping it would have each device tell the other its own files are
//!   somewhere else.
//! * **Derived columns are absent.** EXIF tag rows, DINOv2 embeddings,
//!   sentence vectors, centroids and `representative_*_id` are all recomputed
//!   locally, and the last two are local rowids besides.
//!
//! The one deliberate exception is EXIF: `taken_at`, `make`, `model` and the
//! rest *do* replicate, because a relay servant holds no file to extract them
//! from. Extraction is deterministic, so two devices that both have the
//! original agree anyway and the merge is a no-op.

use serde::{Deserialize, Serialize};

// ── Stamps ──────────────────────────────────────────────────────

/// A row version: hybrid-logical-clock value plus its originating device.
///
/// Ordering is `rev` first, then `rev_dev` as a deterministic tiebreak. The
/// tiebreak is what lets both sides of a sync independently pick the same
/// winner for two edits that happened to land on the same millisecond.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Stamp {
    pub rev: i64,
    pub rev_dev: String,
}

impl Stamp {
    pub fn new(rev: i64, rev_dev: impl Into<String>) -> Self {
        Self {
            rev,
            rev_dev: rev_dev.into(),
        }
    }
}

// ── Entities ────────────────────────────────────────────────────

/// Which replicated table a guid belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Entity {
    Image,
    AiDescription,
    Person,
    FaceDetection,
    Collection,
    CollectionImage,
    Stack,
}

impl Entity {
    /// The SQL table this entity lives in.
    pub fn table(self) -> &'static str {
        match self {
            Entity::Image => "images",
            Entity::AiDescription => "ai_descriptions",
            Entity::Person => "persons",
            Entity::FaceDetection => "face_detections",
            Entity::Collection => "collections",
            Entity::CollectionImage => "collection_images",
            Entity::Stack => "stacks",
        }
    }

    pub fn from_table(table: &str) -> Option<Self> {
        Some(match table {
            "images" => Entity::Image,
            "ai_descriptions" => Entity::AiDescription,
            "persons" => Entity::Person,
            "face_detections" => Entity::FaceDetection,
            "collections" => Entity::Collection,
            "collection_images" => Entity::CollectionImage,
            "stacks" => Entity::Stack,
            _ => return None,
        })
    }
}

/// A row deleted on some device.
///
/// Deletion has to be represented positively: a row that is simply absent is
/// indistinguishable from one the peer has not sent yet, so without this the
/// next sync would helpfully restore everything the user just deleted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tombstone {
    pub guid: String,
    pub entity: Entity,
    #[serde(flatten)]
    pub stamp: Stamp,
}

// ── Rows ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRow {
    pub guid: String,
    #[serde(flatten)]
    pub stamp: Stamp,
    /// BLAKE3 content hash — the only content-stable key an image has, and
    /// the thumbnail/blob cache key. Note it *mutates* on lossless rotation.
    #[serde(with = "hex_hash")]
    pub hash: [u8; 32],
    pub orientation: Option<i64>,
    pub taken_at: Option<i64>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    pub focal_length: Option<f64>,
    pub aperture: Option<f64>,
    pub iso: Option<i64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub stack_guid: Option<String>,

    /// Where the origin device keeps this file. **Advisory only** — applied
    /// when creating a row we have never seen (to name it, and to pick a
    /// destination on a full sync) and never used to overwrite a local path.
    pub origin_path: String,
    /// Advisory, same rule as `origin_path`; lets a relay servant show a size
    /// and lets a transfer plan estimate its cost before fetching anything.
    pub file_size: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AiDescriptionRow {
    pub guid: String,
    #[serde(flatten)]
    pub stamp: Stamp,
    pub image_guid: String,
    pub model_id: String,
    pub description: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonRow {
    pub guid: String,
    #[serde(flatten)]
    pub stamp: Stamp,
    pub name: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceRow {
    pub guid: String,
    #[serde(flatten)]
    pub stamp: Stamp,
    pub image_guid: String,
    /// `[x1, y1, x2, y2]`, normalised to [0, 1].
    pub bbox: [f32; 4],
    /// 512-dim L2-normalised ArcFace vector, base64 of the little-endian f32
    /// blob. Replicated rather than recomputed: re-detecting on the peer
    /// would mint new rows and break every `person_guid` link, and the peer
    /// may not even have the user-supplied ONNX models configured.
    #[serde(with = "b64_blob")]
    pub embedding: Vec<u8>,
    pub confidence: f32,
    pub skipped: bool,
    pub person_guid: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionRow {
    pub guid: String,
    #[serde(flatten)]
    pub stamp: Stamp,
    pub name: String,
    pub color: String,
    pub created_at: i64,
    pub parent_guid: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionImageRow {
    pub guid: String,
    #[serde(flatten)]
    pub stamp: Stamp,
    pub collection_guid: String,
    pub image_guid: String,
    pub added_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StackRow {
    pub guid: String,
    #[serde(flatten)]
    pub stamp: Stamp,
    pub created_at: i64,
    pub cover_image_guid: Option<String>,
}

/// One replicated row of any kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "entity", rename_all = "snake_case")]
pub enum SyncRow {
    Image(ImageRow),
    AiDescription(AiDescriptionRow),
    Person(PersonRow),
    FaceDetection(FaceRow),
    Collection(CollectionRow),
    CollectionImage(CollectionImageRow),
    Stack(StackRow),
}

impl SyncRow {
    pub fn guid(&self) -> &str {
        match self {
            SyncRow::Image(r) => &r.guid,
            SyncRow::AiDescription(r) => &r.guid,
            SyncRow::Person(r) => &r.guid,
            SyncRow::FaceDetection(r) => &r.guid,
            SyncRow::Collection(r) => &r.guid,
            SyncRow::CollectionImage(r) => &r.guid,
            SyncRow::Stack(r) => &r.guid,
        }
    }

    pub fn stamp(&self) -> &Stamp {
        match self {
            SyncRow::Image(r) => &r.stamp,
            SyncRow::AiDescription(r) => &r.stamp,
            SyncRow::Person(r) => &r.stamp,
            SyncRow::FaceDetection(r) => &r.stamp,
            SyncRow::Collection(r) => &r.stamp,
            SyncRow::CollectionImage(r) => &r.stamp,
            SyncRow::Stack(r) => &r.stamp,
        }
    }

    pub fn entity(&self) -> Entity {
        match self {
            SyncRow::Image(_) => Entity::Image,
            SyncRow::AiDescription(_) => Entity::AiDescription,
            SyncRow::Person(_) => Entity::Person,
            SyncRow::FaceDetection(_) => Entity::FaceDetection,
            SyncRow::Collection(_) => Entity::Collection,
            SyncRow::CollectionImage(_) => Entity::CollectionImage,
            SyncRow::Stack(_) => Entity::Stack,
        }
    }
}

/// A decision that two guids name the same photo.
///
/// Replicated rather than recomputed: see the commentary on V19 in
/// `schema.rs`. In short, the two devices hold different sets of
/// duplicate-content rows, so only one of them may be able to see that a
/// merge is unambiguous — and it has to be able to tell the other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuidAlias {
    /// The guid that lost.
    pub alias: String,
    /// The guid now in use.
    pub guid: String,
    #[serde(flatten)]
    pub stamp: Stamp,
}

/// One direction of a sync exchange.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SyncBatch {
    pub rows: Vec<SyncRow>,
    pub tombstones: Vec<Tombstone>,
    #[serde(default)]
    pub aliases: Vec<GuidAlias>,
    /// Highest `rev` covered by this batch — the peer's watermark for the
    /// next pull. Distinct from "the highest rev in `rows`", which would
    /// stall forever if the newest change were a tombstone or a filtered row.
    pub next_rev: i64,
}

impl SyncBatch {
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty() && self.tombstones.is_empty() && self.aliases.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rows.len() + self.tombstones.len() + self.aliases.len()
    }
}

// ── Serde helpers ───────────────────────────────────────────────

/// A 32-byte hash as a lowercase hex string — short enough to eyeball in a
/// log, and a quarter the size of a JSON array of 32 integers.
mod hex_hash {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        let mut out = String::with_capacity(64);
        for b in bytes {
            out.push_str(&format!("{b:02x}"));
        }
        s.serialize_str(&out)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let text = String::deserialize(d)?;
        if text.len() != 64 {
            return Err(serde::de::Error::custom(format!(
                "expected 64 hex characters, got {}",
                text.len()
            )));
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(out)
    }
}

/// An opaque blob as base64. Used for face embeddings: 512 f32s are 2 KB raw,
/// 2.7 KB as base64, but roughly 6 KB written out as a JSON number array.
mod b64_blob {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD
            .decode(text.as_bytes())
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_order_by_rev_then_device() {
        let a = Stamp::new(5, "aaa");
        let b = Stamp::new(5, "bbb");
        let c = Stamp::new(6, "aaa");

        assert!(a < b, "same rev falls back to the device tiebreak");
        assert!(b < c, "a higher rev wins regardless of device");
        // Both sides must agree without coordinating, so the comparison has
        // to be a total order with no ambiguity.
        assert_eq!(a.cmp(&a.clone()), std::cmp::Ordering::Equal);
    }

    #[test]
    fn every_entity_maps_to_a_table_and_back() {
        let all = [
            Entity::Image,
            Entity::AiDescription,
            Entity::Person,
            Entity::FaceDetection,
            Entity::Collection,
            Entity::CollectionImage,
            Entity::Stack,
        ];
        for e in all {
            assert_eq!(Entity::from_table(e.table()), Some(e));
            assert!(
                crate::SYNCED_TABLES.contains(&e.table()),
                "{} is not in SYNCED_TABLES",
                e.table()
            );
        }
        assert_eq!(all.len(), crate::SYNCED_TABLES.len());
        assert_eq!(Entity::from_table("image_exif_tags"), None);
    }

    #[test]
    fn a_hash_survives_a_json_round_trip() {
        let row = ImageRow {
            guid: "g".into(),
            stamp: Stamp::new(1, "dev"),
            hash: [0xab; 32],
            orientation: Some(6),
            taken_at: Some(42),
            make: Some("Fujifilm".into()),
            model: None,
            lens: None,
            focal_length: Some(35.0),
            aperture: None,
            iso: Some(200),
            width: Some(6000),
            height: Some(4000),
            stack_guid: None,
            origin_path: "/photos/a.jpg".into(),
            file_size: 1234,
        };
        let text = serde_json::to_string(&row).expect("serialize");
        assert!(text.contains(&"ab".repeat(32)), "hash should be hex");
        let back: ImageRow = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(row, back);
    }

    #[test]
    fn a_face_embedding_survives_a_json_round_trip() {
        let blob: Vec<u8> = (0..2048).map(|i| (i % 251) as u8).collect();
        let row = FaceRow {
            guid: "g".into(),
            stamp: Stamp::new(9, "dev"),
            image_guid: "img".into(),
            bbox: [0.1, 0.2, 0.3, 0.4],
            embedding: blob.clone(),
            confidence: 0.97,
            skipped: false,
            person_guid: Some("p".into()),
        };
        let text = serde_json::to_string(&row).expect("serialize");
        let back: FaceRow = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.embedding, blob);
        assert_eq!(row, back);
    }

    #[test]
    fn a_row_is_tagged_by_entity_on_the_wire() {
        let row = SyncRow::Person(PersonRow {
            guid: "g".into(),
            stamp: Stamp::new(3, "dev"),
            name: "Ada".into(),
            created_at: 0,
        });
        let text = serde_json::to_string(&row).expect("serialize");
        assert!(text.contains("\"entity\":\"person\""));

        let back: SyncRow = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.entity(), Entity::Person);
        assert_eq!(back.guid(), "g");
        assert_eq!(back.stamp(), &Stamp::new(3, "dev"));
    }

    #[test]
    fn a_malformed_hash_is_rejected_rather_than_truncated() {
        let text = r#"{"guid":"g","rev":1,"rev_dev":"d","hash":"abcd","orientation":null,
            "taken_at":null,"make":null,"model":null,"lens":null,"focal_length":null,
            "aperture":null,"iso":null,"width":null,"height":null,"stack_guid":null,
            "origin_path":"/a.jpg","file_size":0}"#;
        assert!(serde_json::from_str::<ImageRow>(text).is_err());
    }
}
