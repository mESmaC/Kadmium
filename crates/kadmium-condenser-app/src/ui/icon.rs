//! The Condenser app tile, rasterized in code so the window and taskbar icon need no image
//! decoder or bundled bitmap.
//!
//! Geometry matches `assets/condenser-mark.svg`: a cyan C-ring and one short light rod
//! on an abyss rounded tile.

/// Square RGBA pixels of the tile at `size` × `size`, anti-aliased with signed distances.
pub fn tile_rgba(size: u32) -> Vec<u8> {
    const ABYSS: [f32; 3] = [0x08 as f32, 0x0c as f32, 0x14 as f32];
    const GLOW: [f32; 3] = [0x4f as f32, 0xd1 as f32, 0xff as f32];
    const INK: [f32; 3] = [0xe9 as f32, 0xee as f32, 0xf7 as f32];
    const STROKE_RADIUS: f32 = 6.0;

    let scale = 128.0 / size as f32;
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for py in 0..size {
        for px in 0..size {
            let x = (px as f32 + 0.5) * scale;
            let y = (py as f32 + 0.5) * scale;

            let mut color = [0.0f32; 3];
            let mut alpha = coverage(rounded_square(x, y, 64.0, 28.0), scale);
            for (c, a) in color.iter_mut().zip(ABYSS) {
                *c = a;
            }

            let dx = x - 64.0;
            let dy = y - 64.0;
            let ring = ((dx.powi(2) + dy.powi(2)).sqrt() - 34.0).abs() - STROKE_RADIUS;
            let open_arc = if dx > 0.0 && dy.abs() < dx {
                f32::MAX
            } else {
                ring
            };
            let rounded_ends = vertical_capsule(x, y, 88.042, 39.958, 39.958, STROKE_RADIUS).min(
                vertical_capsule(x, y, 88.042, 88.042, 88.042, STROKE_RADIUS),
            );
            blend(
                &mut color,
                &mut alpha,
                GLOW,
                coverage(open_arc.min(rounded_ends), scale),
            );

            let rod = vertical_capsule(x, y, 90.0, 52.0, 76.0, 5.0);
            blend(&mut color, &mut alpha, INK, coverage(rod, scale));

            pixels.extend(color.iter().map(|c| c.round().clamp(0.0, 255.0) as u8));
            pixels.push((alpha * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    pixels
}

/// The tile as a window or taskbar icon.
pub fn window_icon(size: u32) -> Option<tao::window::Icon> {
    tao::window::Icon::from_rgba(tile_rgba(size), size, size).ok()
}

/// Paints `paint` over the current pixel with the given coverage.
fn blend(color: &mut [f32; 3], alpha: &mut f32, paint: [f32; 3], cover: f32) {
    if cover <= 0.0 {
        return;
    }
    let out_alpha = cover + *alpha * (1.0 - cover);
    for (c, p) in color.iter_mut().zip(paint) {
        *c = (p * cover + *c * *alpha * (1.0 - cover)) / out_alpha.max(f32::EPSILON);
    }
    *alpha = out_alpha;
}

/// Pixel coverage for a signed distance (negative inside), with a one-pixel soft edge.
fn coverage(distance: f32, pixel: f32) -> f32 {
    (0.5 - distance / pixel).clamp(0.0, 1.0)
}

/// Signed distance to a square of half-size `half` centred in the tile with corner `radius`.
fn rounded_square(x: f32, y: f32, half: f32, radius: f32) -> f32 {
    let qx = (x - 64.0).abs() - (half - radius);
    let qy = (y - 64.0).abs() - (half - radius);
    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    outside + qx.max(qy).min(0.0) - radius
}

fn vertical_capsule(x: f32, y: f32, cx: f32, top: f32, bottom: f32, radius: f32) -> f32 {
    let clamped = y.clamp(top, bottom);
    ((x - cx).powi(2) + (y - clamped).powi(2)).sqrt() - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(rgba: &[u8], size: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * size + x) * 4) as usize;
        [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
    }

    #[test]
    fn tile_has_transparent_corners_abyss_ground_and_condenser_mark() {
        let size = 128;
        let rgba = tile_rgba(size);
        assert_eq!(rgba.len(), (size * size * 4) as usize);
        assert_eq!(
            pixel(&rgba, size, 0, 0)[3],
            0,
            "rounded corner is transparent"
        );
        assert_eq!(
            pixel(&rgba, size, 20, 110),
            [0x08, 0x0c, 0x14, 255],
            "abyss ground"
        );
        assert_eq!(
            pixel(&rgba, size, 64, 30),
            [0x4f, 0xd1, 0xff, 255],
            "cyan C-ring"
        );
        assert_eq!(
            pixel(&rgba, size, 90, 64),
            [0xe9, 0xee, 0xf7, 255],
            "single light rod"
        );
    }
}
