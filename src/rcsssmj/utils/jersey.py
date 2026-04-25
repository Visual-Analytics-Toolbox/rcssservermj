from typing import Final

import numpy as np

# 5x7 bitmap font for digits 0-9. Each digit is described by 7 rows of 5 bits
# (MSB = leftmost pixel). A set bit denotes a foreground pixel.
_DIGIT_GLYPHS: Final[dict[str, tuple[int, ...]]] = {
    '0': (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    '1': (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    '2': (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    '3': (0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110),
    '4': (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    '5': (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    '6': (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    '7': (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    '8': (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    '9': (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
}

_GLYPH_W: Final[int] = 5
_GLYPH_H: Final[int] = 7


def _glyph_to_array(digit: str) -> np.ndarray:
    """Return a (7, 5) uint8 mask array (1=foreground) for the given digit."""

    rows = _DIGIT_GLYPHS[digit]
    arr = np.zeros((_GLYPH_H, _GLYPH_W), dtype=np.uint8)
    for r, bits in enumerate(rows):
        for c in range(_GLYPH_W):
            if bits & (1 << (_GLYPH_W - 1 - c)):
                arr[r, c] = 1
    return arr


def render_jersey_texture(
    number: int,
    *,
    fg_rgb: tuple[int, int, int] = (255, 255, 255),
    bg_rgb: tuple[int, int, int] = (0, 0, 0),
    width: int = 128,
    height: int = 128,
) -> bytes:
    """Render the given player number into a flat RGB byte buffer.

    The number is rendered centered on a solid background using a 5x7 bitmap
    font, scaled with nearest-neighbor interpolation to maximally fill the
    texture while preserving the glyph aspect ratio.

    Parameter
    ---------
    number: int
        The player number to render (non-negative).

    fg_rgb: tuple[int, int, int]
        Foreground (digit) color as 0-255 RGB triplet.

    bg_rgb: tuple[int, int, int]
        Background color as 0-255 RGB triplet.

    width: int
        Texture width in pixels.

    height: int
        Texture height in pixels.

    Returns
    -------
    bytes
        Tightly packed ``height * width * 3`` RGB byte buffer.
    """

    digits = str(max(0, int(number)))

    # build the unscaled glyph row by horizontally stacking each digit
    # add a 1-pixel gap between digits
    glyph_arrays = [_glyph_to_array(d) for d in digits]
    gap = np.zeros((_GLYPH_H, 1), dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, g in enumerate(glyph_arrays):
        if i > 0:
            pieces.append(gap)
        pieces.append(g)
    glyph_row = np.concatenate(pieces, axis=1) if pieces else np.zeros((_GLYPH_H, _GLYPH_W), dtype=np.uint8)

    gh, gw = glyph_row.shape
    # leave ~10% padding around the glyph
    max_w = max(1, int(width * 0.8))
    max_h = max(1, int(height * 0.8))
    scale = max(1, min(max_w // gw, max_h // gh))

    # nearest-neighbor upscale of the glyph row
    scaled = np.repeat(np.repeat(glyph_row, scale, axis=0), scale, axis=1)
    sh, sw = scaled.shape

    # compose onto the canvas
    bg = np.array(bg_rgb, dtype=np.uint8)
    fg = np.array(fg_rgb, dtype=np.uint8)
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[..., :] = bg

    y0 = (height - sh) // 2
    x0 = (width - sw) // 2
    region = canvas[y0:y0 + sh, x0:x0 + sw]
    mask = scaled.astype(bool)
    region[mask] = fg

    return canvas.tobytes()
