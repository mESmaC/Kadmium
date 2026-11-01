//! Differential check against the Lean model of FTS3 chunk manifests in
//! `formal/KadmiumFormal/Data/Fts3Chunks.lean`, which proves the chunks tile the bank
//! contiguously. This fixture, written by `lake exe kadmium_vectors`, ties the model to
//! `Fts3ChunkSlicer::compute_chunk_manifest`.

use kadmium_data::transcoder::Fts3ChunkSlicer;

const VECTORS: &str = include_str!("fixtures/fts3_chunks_lean_vectors.txt");

#[test]
fn chunk_manifests_match_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "chunks");
        let len: usize = f[1].parse().expect("length");
        let requested: usize = f[2].parse().expect("requested chunk size");
        let count: usize = f[3].parse().expect("count");
        let entries: Vec<(u64, u64)> = if f[4] == "-" {
            Vec::new()
        } else {
            f[4].split(',')
                .map(|e| {
                    let (o, l) = e.split_once(':').expect("offset:len");
                    (o.parse().expect("offset"), l.parse().expect("len"))
                })
                .collect()
        };
        let bytes = vec![0u8; len];
        let manifest = Fts3ChunkSlicer::compute_chunk_manifest(&bytes, requested);
        assert_eq!(manifest.len(), count, "chunk count differs for {line}");
        for (i, (desc, (offset, chunk_len))) in manifest.iter().zip(&entries).enumerate() {
            assert_eq!(desc.chunk_index as usize, i);
            assert_eq!(desc.byte_offset, *offset, "offset of chunk {i} in {line}");
            assert_eq!(desc.byte_len, *chunk_len, "length of chunk {i} in {line}");
        }
        checked += 1;
    }
    assert!(checked >= 9, "regenerate with `lake exe kadmium_vectors`");
}
