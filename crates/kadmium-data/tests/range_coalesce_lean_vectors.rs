//! Differential check against the Lean model of HTTP range coalescing in
//! `formal/KadmiumFormal/Data/RangeCoalesce.lean`, which proves the merged requests partition the
//! slices, contain them, are more than `max_gap` apart, fetch no run of more than `max_gap`
//! unrequested bytes, and are as few as any plan with that property. This fixture, written by
//! `lake exe kadmium_vectors`, ties the model to `HfByteRangeSlicer::coalesce_range_requests`.

use kadmium_data::{HfByteRangeSlicer, HttpByteRangeSlice, TensorEncoding};

const VECTORS: &str = include_str!("fixtures/range_coalesce_lean_vectors.txt");

struct Expected {
    start: u64,
    end: u64,
    members: Vec<usize>,
    gap: u64,
}

fn slice(url: &str, start: u64, end: u64) -> HttpByteRangeSlice {
    HttpByteRangeSlice {
        tensor_name: format!("{url}:{start}"),
        layer_index: 0,
        expert_index: None,
        shard_filename: "shard".to_string(),
        url: url.to_string(),
        http_range_header: format!("bytes={start}-{end}"),
        byte_start: start,
        byte_end_inclusive: end,
        payload_len_bytes: (end - start + 1) as usize,
        target_encoding: TensorEncoding::Fp32,
    }
}

/// One fixture line: the gap, the slices as `(start, stop)` and the expected requests.
type Case = (u64, Vec<(u64, u64)>, Vec<Expected>);

fn parse(line: &str) -> Case {
    let f: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(f[0], "plan");
    let gap: u64 = f[1].parse().expect("gap");
    let slices: Vec<(u64, u64)> = if f[2] == "-" {
        Vec::new()
    } else {
        f[2].split(',')
            .map(|s| {
                let (a, b) = s.split_once('-').expect("start-stop");
                (a.parse().expect("start"), b.parse().expect("stop"))
            })
            .collect()
    };
    let requests: Vec<Expected> = if f[3] == "-" {
        Vec::new()
    } else {
        f[3..]
            .iter()
            .map(|r| {
                let (range, rest) = r.split_once(':').expect("range:members;gap");
                let (a, b) = range.split_once('-').expect("start-stop");
                let (members, gap) = rest.split_once(';').expect("members;gap");
                Expected {
                    start: a.parse().expect("start"),
                    end: b.parse().expect("end"),
                    members: members
                        .split(',')
                        .map(|m| m.parse().expect("member"))
                        .collect(),
                    gap: gap.parse().expect("gap"),
                }
            })
            .collect()
    };
    (gap, slices, requests)
}

fn cases() -> Vec<Case> {
    VECTORS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(parse)
        .collect()
}

#[test]
fn one_url_plans_match_the_lean_model() {
    let cases = cases();
    assert!(
        cases.len() >= 8,
        "regenerate with `lake exe kadmium_vectors`"
    );
    for (gap, slices, expected) in &cases {
        let input: Vec<HttpByteRangeSlice> =
            slices.iter().map(|&(s, e)| slice("u", s, e)).collect();
        let plan = HfByteRangeSlicer::coalesce_range_requests(&input, *gap);
        assert_eq!(
            plan.len(),
            expected.len(),
            "request count differs for gap {gap} {slices:?}"
        );
        for (req, exp) in plan.iter().zip(expected) {
            assert_eq!(req.byte_start, exp.start);
            assert_eq!(req.byte_end_inclusive, exp.end);
            assert_eq!(req.members, exp.members);
            assert_eq!(req.gap_bytes, exp.gap);
            assert_eq!(
                req.http_range_header,
                format!("bytes={}-{}", exp.start, exp.end)
            );
        }
    }
}

/// Two URLs interleaved in the input: the Rust sorts by URL first, so the result is the model's
/// plan for each URL in turn, with members pointing back at the interleaved positions.
#[test]
fn urls_never_merge_and_each_follows_the_model() {
    for (gap, slices, expected) in &cases() {
        if slices.is_empty() {
            continue;
        }
        let mut input = Vec::new();
        for &(s, e) in slices {
            input.push(slice("a", s, e));
            input.push(slice("b", s, e));
        }
        let plan = HfByteRangeSlicer::coalesce_range_requests(&input, *gap);
        assert_eq!(plan.len(), 2 * expected.len());
        for (k, url) in ["a", "b"].iter().enumerate() {
            for (req, exp) in plan[k * expected.len()..(k + 1) * expected.len()]
                .iter()
                .zip(expected)
            {
                assert_eq!(req.url, *url);
                assert_eq!(req.byte_start, exp.start);
                assert_eq!(req.byte_end_inclusive, exp.end);
                let members: Vec<usize> = exp.members.iter().map(|m| 2 * m + k).collect();
                assert_eq!(req.members, members);
            }
        }
    }
}
