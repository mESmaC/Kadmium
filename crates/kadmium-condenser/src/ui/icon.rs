//! The Kadmium app tile, rasterized in code so the window and taskbar icon need no image
//! decoder or bundled bitmap.
//!
//! Geometry matches `design-system/assets/Logos/kadmium-tile-graphite.svg` (a 128-unit
//! graphite tile, four cadmium control rods at unequal depths and a light head plate).

/// Square RGBA pixels of the tile at `size` × `size`, anti-aliased with signed distances.
pub fn tile_rgba(size: u32) -> Vec<u8> {
    const GRAPHITE: [f32; 3] = [0x16 as f32, 0x19 as f32, 0x1e as f32];
    const CADMIUM: [f32; 3] = [0xf3 as f32, 0xc0 as f32, 0x20 as f32];
    const PLATE: [f32; 3] = [0xec as f32, 0xef as f32, 0xf3 as f32];
    // (x, y_top, y_bottom) of each rod in the 128-unit tile.
    const RODS: [(f32, f32, f32); 4] = [
        (46.0, 37.6, 73.6),
        (58.0, 37.6, 90.4),
        (70.0, 37.6, 61.6),
        (82.0, 37.6, 80.8),
    ];
    const STROKE_RADIUS: f32 = 4.2;

    let scale = 128.0 / size as f32;
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for py in 0..size {
        for px in 0..size {
            let x = (px as f32 + 0.5) * scale;
            let y = (py as f32 + 0.5) * scale;

            let mut color = [0.0f32; 3];
            let mut alpha = coverage(rounded_square(x, y, 64.0, 28.0), scale);
            for (c, g) in color.iter_mut().zip(GRAPHITE) {
                *c = g;
            }

            let rods = RODS
                .iter()
                .map(|&(rx, top, bottom)| vertical_capsule(x, y, rx, top, bottom, STROKE_RADIUS))
                .fold(f32::MAX, f32::min);
            blend(&mut color, &mut alpha, CADMIUM, coverage(rods, scale));

            let plate = horizontal_capsule(x, y, 48.4, 38.8, 89.2, STROKE_RADIUS);
            blend(&mut color, &mut alpha, PLATE, coverage(plate, scale));

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

fn horizontal_capsule(x: f32, y: f32, cy: f32, left: f32, right: f32, radius: f32) -> f32 {
    let clamped = x.clamp(left, right);
    ((x - clamped).powi(2) + (y - cy).powi(2)).sqrt() - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(rgba: &[u8], size: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * size + x) * 4) as usize;
        [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
    }

    #[test]
    fn tile_has_transparent_corners_graphite_ground_and_cadmium_rods() {
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
            [0x16, 0x19, 0x1e, 255],
            "graphite ground"
        );
        assert_eq!(
            pixel(&rgba, size, 58, 80),
            [0xf3, 0xc0, 0x20, 255],
            "deepest rod"
        );
        assert_eq!(
            pixel(&rgba, size, 64, 48),
            [0xec, 0xef, 0xf3, 255],
            "head plate"
        );
    }
}
