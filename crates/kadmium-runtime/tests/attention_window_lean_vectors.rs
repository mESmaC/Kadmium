//! Differential check of the Lean attention-mask bounds against the decoder's actual method.

use kadmium_runtime::native_decoder::DecoderConfig;

#[test]
fn attention_window_lean_vectors() {
    let mut config = DecoderConfig::from_json_str(include_str!(
        "fixtures/native_decoder/mistral_f16/config.json"
    ))
    .unwrap();
    let mut checked = 0;
    for line in include_str!("fixtures/attention_window_lean_vectors.txt").lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let values: Vec<u64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(values.len(), 4);
        // Include the usize boundary on 64-bit hosts, and retain the smaller vectors on 32-bit.
        let Ok(values) = values
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
        else {
            continue;
        };
        config.sliding_window = (values[0] != 0).then_some(values[0]);
        assert_eq!(config.attention_range(values[1]), values[2]..values[3]);
        checked += 1;
    }
    assert!(checked >= 50);
}
