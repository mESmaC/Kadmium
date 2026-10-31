//! MoE router selection and attention row masks.
//!
//! The top-k order is InferSSD's deterministic warp top-k (`ggml-cuda/top-k.cu`): a NaN logit
//! scores as `-f32::MAX`, and equal scores rank by the lower expert id, so the selection and
//! its order are a function of the logits alone. Kadmium's synthetic router sorts with the same
//! rule (a stable descending sort keeps equal scores in id order).
//!
//! Where InferSSD clears a per-lane bit for each winner, each round here instead admits only the
//! candidates ranked after the previous winner. The rule needs no state beyond the last winner,
//! so it works for any expert count rather than a compile-time column count.

/// The score a logit competes with: NaN becomes the lowest finite value.
#[inline(always)]
pub fn router_score(logit: f32) -> f32 {
    if logit.is_nan() { -f32::MAX } else { logit }
}

/// Whether expert `a` (score, id) is ranked before expert `b`: higher score first, then lower id.
#[inline(always)]
pub fn ranks_before(a: (f32, u32), b: (f32, u32)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// The better ranked of two candidates, `None` meaning no candidate.
#[inline(always)]
pub fn better(a: Option<(f32, u32)>, b: Option<(f32, u32)>) -> Option<(f32, u32)> {
    match (a, b) {
        (Some(x), Some(y)) => Some(if ranks_before(y, x) { y } else { x }),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Whether a candidate is still eligible after `previous` was selected.
#[inline(always)]
pub fn eligible(candidate: (f32, u32), previous: Option<(f32, u32)>) -> bool {
    match previous {
        None => true,
        Some(previous) => ranks_before(previous, candidate),
    }
}

/// Additive attention mask entry: `0.0` for a visible column, `-inf` past the visible count.
#[inline(always)]
pub fn row_mask_value(column: u32, visible: i32) -> f32 {
    if (column as i64) < visible as i64 {
        0.0
    } else {
        f32::NEG_INFINITY
    }
}
