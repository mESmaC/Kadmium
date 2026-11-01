//! Differential check against the Lean model of KDM2 framing in
//! `formal/KadmiumFormal/Network/WireFrame.lean`.
//!
//! The theorems there cover every legal frame; this fixture, written by `lake exe
//! kadmium_vectors`, ties the model to `WireFrame::encode` and `WireFrame::decode` on a sample of
//! frames, legacy `KDMF` frames and malformed inputs.

use kadmium_network::{WireFrame, WireFrameKind};

const VECTORS: &str = include_str!("fixtures/wire_frame_lean_vectors.txt");

fn hex(s: &str) -> Vec<u8> {
    if s == "-" {
        return Vec::new();
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex byte"))
        .collect()
}

fn lines() -> impl Iterator<Item = Vec<&'static str>> {
    VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect())
}

#[test]
fn fixture_has_every_section() {
    let kinds: Vec<&str> = lines().map(|f| f[0]).collect();
    assert!(kinds.contains(&"frame"));
    assert!(kinds.contains(&"legacy"));
    assert!(kinds.contains(&"reject"));
}

/// `read_from_stream` parses the same header as `decode` and only then reads the payload, so it
/// agrees with the decoder on every fixture frame and refuses every rejected input, including the
/// oversize length, without reaching a payload byte.
#[test]
fn stream_reader_agrees_with_the_decoder() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        for f in lines().filter(|f| f[0] == "frame" || f[0] == "legacy") {
            let encoded = hex(if f[0] == "frame" { f[5] } else { f[4] });
            let mut cursor: &[u8] = &encoded;
            let streamed = WireFrame::read_from_stream(&mut cursor)
                .await
                .expect("the stream reader accepts every fixture frame");
            assert_eq!(streamed, WireFrame::decode(&encoded).expect("decodes"));
            assert!(cursor.is_empty(), "the reader consumed exactly one frame");
        }
        for f in lines().filter(|f| f[0] == "reject") {
            let input = hex(f[1]);
            let mut cursor: &[u8] = &input;
            assert!(
                WireFrame::read_from_stream(&mut cursor).await.is_err(),
                "the stream reader accepted {} which the model rejects as {}",
                f[1],
                f[2]
            );
        }
    });
}

#[test]
fn frames_encode_and_decode_as_the_lean_model() {
    for f in lines().filter(|f| f[0] == "frame") {
        let kind = WireFrameKind::from_u8(f[1].parse().expect("kind")).expect("legal kind");
        let epoch: u32 = f[2].parse().expect("epoch");
        let request_id: u64 = f[3].parse().expect("request id");
        let payload = hex(f[4]);
        let encoded = hex(f[5]);
        let frame = WireFrame::new(kind, epoch, payload).with_request_id(request_id);
        assert_eq!(
            frame.encode(),
            encoded,
            "encode differs for {}",
            f[1..4].join(" ")
        );
        assert_eq!(WireFrame::decode(&encoded).expect("decodes"), frame);
        let mut longer = encoded.clone();
        longer.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        assert_eq!(
            WireFrame::decode(&longer).expect("decodes a prefix"),
            frame,
            "trailing bytes change the decoded frame"
        );
    }
}

#[test]
fn legacy_frames_decode_with_request_id_zero() {
    for f in lines().filter(|f| f[0] == "legacy") {
        let kind: u8 = f[1].parse().expect("kind");
        let epoch: u32 = f[2].parse().expect("epoch");
        let payload = hex(f[3]);
        let decoded = WireFrame::decode(&hex(f[4])).expect("legacy frame decodes");
        assert_eq!(decoded.kind as u8, kind);
        assert_eq!(decoded.epoch, epoch);
        assert_eq!(decoded.request_id, 0);
        assert_eq!(decoded.payload, payload);
    }
}

#[test]
fn malformed_frames_are_rejected_for_the_reason_the_model_gives() {
    for f in lines().filter(|f| f[0] == "reject") {
        let input = hex(f[1]);
        let err = WireFrame::decode(&input).expect_err("the model rejects this input");
        let expected = match f[2] {
            "short_magic" => "shorter than 4-byte magic",
            "bad_magic" => "Invalid WireFrame magic",
            "short_header" => "-byte header",
            "bad_kind" => "Unknown WireFrameKind",
            "oversize" => "exceeds",
            "truncated" => "truncated",
            other => panic!("unknown rejection {other}"),
        };
        assert!(
            err.contains(expected),
            "input {} rejected as {err:?}, model says {}",
            f[1],
            f[2]
        );
    }
}
