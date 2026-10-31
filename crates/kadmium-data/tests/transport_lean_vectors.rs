//! Differential check against the Lean model of transport chunk maps in `formal/`.
//!
//! `formal/KadmiumFormal/Transport/ChunkMap.lean` proves that the sender's read loop produces
//! exactly the chunk map the receiver's `validate` accepts, and that this map tiles the file.
//! Those theorems are about a Lean model of this code. This test ties the model to the code:
//! `lake exe transport_vectors` evaluates the model and writes
//! `tests/fixtures/transport_chunk_vectors.txt`, and both `chunk_plan` (the receiver's check)
//! and `hash_stream` (the sender's read loop) must reproduce every map exactly.

use kadmium_data::checkpoint::transport::{chunk_plan, hash_stream};
use std::io::Read;

const VECTORS: &str = include_str!("fixtures/transport_chunk_vectors.txt");

/// A stream of `len` bytes that returns at most `step` bytes per read, so the sender's loop is
/// exercised with short reads as well as full ones.
struct Trickle {
    len: u64,
    step: usize,
}

impl Read for Trickle {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = (self.len.min(self.step as u64) as usize).min(buf.len());
        buf[..n].fill(0x5a);
        self.len -= n as u64;
        Ok(n)
    }
}

#[test]
fn chunk_maps_match_the_lean_model() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let mut fields = line.split(' ');
        let bytes: u64 = fields.next().unwrap().parse().unwrap();
        let chunk: u64 = fields.next().unwrap().parse().unwrap();
        let expected: Vec<(u64, u64)> = fields
            .map(|pair| {
                let (offset, len) = pair.split_once(':').unwrap();
                (offset.parse().unwrap(), len.parse().unwrap())
            })
            .collect();

        assert_eq!(
            chunk_plan(bytes, chunk),
            expected,
            "chunk_plan({bytes}, {chunk})"
        );
        for step in [1 << 20, 3] {
            // A 3-byte trickle through a 16 MiB chunk is needlessly slow; full reads cover it.
            if step == 3 && bytes > 1 << 16 {
                continue;
            }
            let read = hash_stream(Trickle { len: bytes, step }, "f", chunk).unwrap();
            let got: Vec<(u64, u64)> = read.chunks.iter().map(|c| (c.offset, c.bytes)).collect();
            assert_eq!(
                got, expected,
                "hash_stream({bytes}, {chunk}) with {step}-byte reads"
            );
            assert_eq!(read.bytes, bytes);
        }
        checked += 1;
    }
    assert!(checked >= 90, "fixture has only {checked} vectors");
}
