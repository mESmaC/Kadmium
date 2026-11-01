//! Differential check against the Lean model of the `.ktb` bank layout in
//! `formal/KadmiumFormal/Data/KtbLayout.lean`, which proves where every tensor lands, that the
//! reservations never overlap, that every byte is copied unchanged and that the reader accepts
//! what the packer wrote. This fixture, written by `lake exe kadmium_vectors`, ties the model to
//! `KtbBankPacker::pack_to_bytes` and `KtbBankReader`.

use kadmium_data::{
    KTB_PAGE_ALIGNMENT_BYTES, KtbBankPacker, KtbBankReader, TensorEncoding,
    compute_fnv1a64_checksum,
};

const VECTORS: &str = include_str!("fixtures/ktb_layout_lean_vectors.txt");

/// SplitMix64 of `i + 1`, as `splitmix` in `formal/Fixtures.lean`.
fn splitmix(i: u64) -> u64 {
    let mut z = i.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `byteAt` in `formal/Fixtures.lean`.
fn byte_at(seed: u64, j: u64) -> u8 {
    (splitmix(seed * 1_000_003 + j) % 256) as u8
}

fn nats(s: &str) -> Vec<u64> {
    if s == "-" {
        return Vec::new();
    }
    s.split(',').map(|n| n.parse().expect("number")).collect()
}

#[test]
fn packed_banks_have_the_layout_and_bytes_the_lean_model_gives() {
    let mut checked = 0;
    for line in VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f[0], "layout");
        let seed: u64 = f[1].parse().expect("seed");
        let sizes = nats(f[2]);
        let offsets = nats(f[3]);
        let padded = nats(f[4]);
        let total: usize = f[5].parse().expect("total");
        let digest = u64::from_str_radix(f[6], 16).expect("digest");

        let tensors: Vec<Vec<u8>> = sizes
            .iter()
            .enumerate()
            .map(|(i, &len)| (0..len).map(|j| byte_at(seed + i as u64, j)).collect())
            .collect();
        let mut packer = KtbBankPacker::new("lean-layout");
        for (i, data) in tensors.iter().enumerate() {
            packer.add_tensor_bank(
                format!("t{i}"),
                i as u32,
                None,
                TensorEncoding::Fp32,
                data.clone(),
            );
        }
        let image = packer.pack_to_bytes().expect("packs");
        assert_eq!(image.len(), total, "bank length differs for {line}");
        assert_eq!(
            compute_fnv1a64_checksum(&image[KTB_PAGE_ALIGNMENT_BYTES as usize..]),
            digest,
            "bytes after page 0 differ for {line}"
        );

        let reader =
            KtbBankReader::from_slice(&image).expect("the reader accepts what the packer wrote");
        assert_eq!(reader.header.entries.len(), sizes.len());
        for (i, entry) in reader.header.entries.iter().enumerate() {
            assert_eq!(
                entry.page_offset_bytes, offsets[i],
                "offset of tensor {i} in {line}"
            );
            assert_eq!(
                entry.padded_reservation_bytes, padded[i],
                "reservation of tensor {i}"
            );
            assert_eq!(entry.payload_bytes, sizes[i]);
            let slice = reader
                .get_verified_tensor_slice(entry)
                .expect("every packed entry passes tensor_range and its checksum");
            assert_eq!(slice, tensors[i].as_slice(), "tensor {i} bytes changed");
        }
        checked += 1;
    }
    assert!(checked >= 7, "regenerate with `lake exe kadmium_vectors`");
}
