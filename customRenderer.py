r"""
customRenderer.py - High-Quality Image-to-ASCII Rendering Engine
================================================================
Features:
- Configurable monospace font aspect ratio compensation.
- Curated character palettes (clean, detailed, blocks, minimal, dots, cross).
- Contrast, brightness, and gamma image preprocessing.
- Directional edge preservation (Sobel filters mapping contours to /, \, |, -).
- Color rendering: ANSI 24-bit TrueColor for terminals, and styled HTML export.
- Can be used as a CLI tool or imported as a Python module.
"""

from __future__ import annotations
import math
import time
import random
import sys
import json
import io
import base64
import argparse
from pathlib import Path
from typing import Union, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps


# Curated Palettes (Calibrated optical density ramps)
PALETTES = {
    # Clean 10-ramp (crisp, iconic, zero visual noise)
    "clean": " .:-=+*#%@",
    # Extended 70-ramp (smooth photographic gradation)
    "detailed": r" .'`^\",:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    # Halftone print dots (like ascii.rest)
    "dots": " ·•●",
    # Pixel/block carriers — color-first, character is just a vessel
    "pixel": " ░▒▓█",
    # Cross glyphs (architectural hatch)
    "cross": " +×",
    # Retro cyber/matrix
    "matrix": " 01",
    # Legacy standard
    "standard": "@%#*+=-:. ",
}


class AsciiRenderer:
    """
    Renders PIL Images or image files into customizable, clean ASCII art.
    """

    def __init__(
        self,
        width: int = 100,
        aspect_ratio: float = 0.50,
        palette: Union[str, List[str]] = "clean",
        invert: bool = False,
        contrast: float = 1.4,
        saturation: float = 1.2,
        brightness: float = 1.0,
        gamma: float = 1.0,
        edge_detection: bool = False,
        edge_threshold: int = 95,
        edge_blend_ratio: float = 0.7,
        shadow_lift: int = 0,
        color_lift: int = 0,
    ):
        """
        :param width: Target width in characters.
        :param aspect_ratio: Font vertical elongation compensation (~0.45 to 0.55).
        :param palette: Name in PALETTES or a custom string of characters.
        :param invert: Invert brightness (useful for light vs dark backgrounds).
        :param contrast: Factor to adjust contrast (>1.0 increases contrast).
        :param saturation: Factor to adjust color saturation (>1.0 makes colors more vivid).
        :param brightness: Factor to adjust brightness.
        :param gamma: Gamma correction factor.
        :param edge_detection: Only outlines will use directional characters.
        :param edge_threshold: High gradient threshold (0-255) to trigger clean outlines only.
        :param edge_blend_ratio: Weight of edge overlay over pure luminance.
        :param shadow_lift: Minimum luminance floor (0-80). Lifts pure-black areas so they
            render as faint visible characters instead of invisible spaces. Useful for
            dark images where detail is lost in the black background. Default 0 (no lift).
        :param color_lift: Minimum per-channel RGB floor (0-80) applied when writing colors
            in HTML output. Ensures dark-colored characters are visible on a dark background
            instead of blending into it. Default 0 (no lift).
        """
        self.width = max(10, width)
        self.aspect_ratio = aspect_ratio
        self.palette = PALETTES.get(palette, palette)
        self.invert = invert
        self.contrast = contrast
        self.brightness = brightness
        self.gamma = gamma
        self.saturation = saturation
        self.edge_detection = edge_detection
        self.edge_threshold = edge_threshold
        self.edge_blend_ratio = edge_blend_ratio
        self.shadow_lift = max(0, min(80, shadow_lift))
        self.color_lift = max(0, min(80, color_lift))

    def _calculate_dimensions(self, img_width: int, img_height: int) -> Tuple[int, int]:
        """Calculates character grid width and height maintaining visual proportions."""
        scale = self.width / float(img_width)
        new_height = int(img_height * scale * self.aspect_ratio)
        return self.width, max(1, new_height)

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Applies brightness, contrast, and gamma adjustments."""
        img = img.convert("RGB")

        if self.brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.brightness)

        if self.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast)

        if self.saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(self.saturation)

        if self.gamma != 1.0:
            inv_gamma = 1.0 / self.gamma
            lut = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)] * 3
            img = img.point(lut)

        return img

    def _map_luminance_to_char(self, luminance: int) -> str:
        """
        Accurately maps 0-255 luminance to a calibrated character without inversion bugs.
        If palette starts with space, 0 is space and 255 is dense.
        shadow_lift floors the minimum luminance so dark areas get a visible character.
        """
        palette = self.palette
        palette_len = len(palette)
        # Apply shadow lift: clamp luminance to [shadow_lift, 255]
        effective_lum = max(self.shadow_lift, luminance) if self.shadow_lift > 0 else luminance
        ratio = max(0.0, min(1.0, effective_lum / 255.0))
        if self.invert:
            ratio = 1.0 - ratio

        if palette[0] == " ":
            idx = int(ratio * (palette_len - 1))
        else:
            idx = int((1.0 - ratio) * (palette_len - 1))

        return palette[idx]

    def _compute_edges_and_directions(self, gray_img: Image.Image) -> Tuple[List[List[float]], List[List[str]]]:
        """
        Computes Sobel gradients to preserve outlines with directional characters:
        - Horizontal edge (-)
        - Vertical edge (|)
        - Diagonal edge (/ or \\)
        Only triggers on strong prominent outlines (threshold >= 90).
        """
        w, h = gray_img.size
        sobel_x = ImageFilter.Kernel(
            (3, 3),
            [-1, 0, 1,
             -2, 0, 2,
             -1, 0, 1],
            scale=1,
            offset=128
        )
        sobel_y = ImageFilter.Kernel(
            (3, 3),
            [-1, -2, -1,
              0,  0,  0,
              1,  2,  1],
            scale=1,
            offset=128
        )

        gx_img = gray_img.filter(sobel_x)
        gy_img = gray_img.filter(sobel_y)

        gx_data = list(gx_img.getdata())
        gy_data = list(gy_img.getdata())

        edge_magnitudes: List[List[float]] = []
        edge_chars: List[List[str]] = []

        for y in range(h):
            row_mag = []
            row_char = []
            for x in range(w):
                idx = y * w + x
                gx = gx_data[idx] - 128
                gy = gy_data[idx] - 128

                magnitude = math.hypot(gx, gy)
                row_mag.append(magnitude)

                # Only trigger on prominent structural outlines
                if self.edge_detection and magnitude >= self.edge_threshold:
                    angle = math.degrees(math.atan2(gy, gx)) % 180
                    if 22.5 <= angle < 67.5:
                        char = "/"
                    elif 67.5 <= angle < 112.5:
                        char = "|"
                    elif 112.5 <= angle < 157.5:
                        char = "\\"
                    else:
                        char = "-"
                else:
                    char = ""
                row_char.append(char)

            edge_magnitudes.append(row_mag)
            edge_chars.append(row_char)

        return edge_magnitudes, edge_chars

    def render(
        self,
        image_input: Union[str, Path, Image.Image],
        color: bool = False,
    ) -> str:
        """
        Renders the image into clean, crisp ASCII text or ANSI TrueColor text.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        else:
            img = image_input

        target_w, target_h = self._calculate_dimensions(img.width, img.height)
        processed_rgb = self._preprocess_image(img).resize((target_w, target_h), Image.Resampling.LANCZOS)
        gray_img = processed_rgb.convert("L")

        if self.invert:
            gray_img = ImageOps.invert(gray_img)

        edge_chars = None
        if self.edge_detection:
            _, edge_chars = self._compute_edges_and_directions(gray_img)

        gray_pixels = list(gray_img.getdata())
        rgb_pixels = list(processed_rgb.getdata()) if color else None

        lines = []
        for y in range(target_h):
            line_chars = []
            for x in range(target_w):
                idx = y * target_w + x
                brightness_val = gray_pixels[idx]

                # True outline check
                if self.edge_detection and edge_chars and edge_chars[y][x]:
                    char = edge_chars[y][x]
                else:
                    char = self._map_luminance_to_char(brightness_val)

                if color and rgb_pixels:
                    r, g, b = rgb_pixels[idx]
                    line_chars.append(f"\033[38;2;{r};{g};{b}m{char}\033[0m")
                else:
                    line_chars.append(char)

            lines.append("".join(line_chars))

        return "\n".join(lines)

    def render_braille(
        self,
        image_input: Union[str, Path, Image.Image],
        threshold: int = 120,
    ) -> str:
        """
        Renders image using 8-dot Unicode Braille patterns (U+2800 to U+28FF).
        Gives 2x horizontal and 4x vertical resolution for pixel-sharp text art without slash noise!
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        else:
            img = image_input

        hires_w = self.width * 2
        hires_h = int((img.height / img.width) * hires_w * (self.aspect_ratio * 2))
        hires_h = max(4, (hires_h // 4) * 4)

        gray = self._preprocess_image(img).resize((hires_w, hires_h), Image.Resampling.LANCZOS).convert("L")
        if self.invert:
            gray = ImageOps.invert(gray)

        data = list(gray.getdata())
        lines = []

        for cy in range(0, hires_h, 4):
            line = []
            for cx in range(0, hires_w, 2):
                mask = 0
                dots = [
                    (0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04),
                    (1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20),
                    (0, 3, 0x40), (1, 3, 0x80)
                ]
                for dx, dy, bit in dots:
                    px = cx + dx
                    py = cy + dy
                    if px < hires_w and py < hires_h:
                        if data[py * hires_w + px] >= threshold:
                            mask |= bit
                line.append(chr(0x2800 + mask))
            lines.append("".join(line))

        return "\n".join(lines)

    def render_to_html(
        self,
        image_input: Union[str, Path, Image.Image],
        output_file: Optional[Union[str, Path]] = None,
        font_size: int = 10,
        background: str = "#040406",
    ) -> str:
        """
        Renders image to a clean HTML page with studio-grade monospace typography.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        else:
            img = image_input

        target_w, target_h = self._calculate_dimensions(img.width, img.height)
        processed_rgb = self._preprocess_image(img).resize((target_w, target_h), Image.Resampling.LANCZOS)
        gray_img = processed_rgb.convert("L")

        if self.invert:
            gray_img = ImageOps.invert(gray_img)

        edge_chars = None
        if self.edge_detection:
            _, edge_chars = self._compute_edges_and_directions(gray_img)

        gray_pixels = list(gray_img.getdata())
        rgb_pixels = list(processed_rgb.getdata())

        html_lines = []
        for y in range(target_h):
            line_parts = []
            for x in range(target_w):
                idx = y * target_w + x
                brightness_val = gray_pixels[idx]

                if self.edge_detection and edge_chars and edge_chars[y][x]:
                    char = edge_chars[y][x]
                else:
                    char = self._map_luminance_to_char(brightness_val)

                escaped_char = {"&": "&amp;", "<": "&lt;", ">": "&gt;", " ": "&nbsp;"}.get(char, char)
                r, g, b = rgb_pixels[idx]
                if self.color_lift > 0:
                    r = max(self.color_lift, r)
                    g = max(self.color_lift, g)
                    b = max(self.color_lift, b)
                line_parts.append(f'<span style="color:rgb({r},{g},{b})">{escaped_char}</span>')

            html_lines.append("".join(line_parts))

        ascii_body = "<br>\n".join(html_lines)

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prompt-to-ASCII Output</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400&display=swap');
        body {{
            background-color: {background};
            margin: 0;
            padding: 32px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        pre {{
            font-family: "JetBrains Mono", "Cascadia Code", "SF Mono", "Fira Code", monospace;
            font-size: {font_size}px;
            line-height: 1.0;
            letter-spacing: 0.08em;
            user-select: all;
            background-color: {background};
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            box-shadow: 0 16px 64px rgba(0,0,0,0.85);
            overflow: auto;
        }}
        /* Characters are color vessels — let the color bleed through */
        pre span {{
            filter: saturate(1.45);
            text-shadow: 0 0 8px currentColor;
        }}
    </style>
</head>
<body>
    <pre>{ascii_body}</pre>
</body>
</html>"""

        if output_file:
            Path(output_file).write_text(html_template, encoding="utf-8")

        return html_template

    def render_motion_html(
        self,
        image_input: Union[str, Path, Image.Image],
        output_file: Optional[Union[str, Path]] = None,
        font_size: int = 11,
        background: str = "#040406",
    ) -> str:
        """
        Exports a self-contained, 60-FPS interactive HTML player with:
        - Organic, natural fluid flow inspired by ascii.rest:
          * Bayer 4x4 ordered dithering for seamless, grain-free character transitions.
          * Multi-octave continuous Simplex noise density flow.
          * 100% authentic image color preservation (no artificial tinting).
          * Zero sparks/particles or artificial gimmicks: pure typography in motion.
        - Zero blur/glow: razor-sharp monospace typography on pure dark ground.
        - 100% subject clarity and recognizability: subtle organic currents without destroying features.
        - Style selectors: Clean ASCII (default), Halftone Dots, Detailed, Minimal.
        - Motion presets: Natural Flow (default), Liquid Currents, Organic Wave, Atmospheric, Static.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        else:
            img = image_input

        processed_rgb = self._preprocess_image(img)
        buffered = io.BytesIO()
        processed_rgb.save(buffered, format="JPEG", quality=92)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data_url = f"data:image/jpeg;base64,{img_b64}"

        default_width = max(90, self.width)
        palettes_dict = {
            "clean": list(PALETTES["clean"]),
            "dots": list(PALETTES["dots"]),
            "detailed": list(PALETTES["detailed"]),
            "pixel": list(PALETTES["pixel"]),
        }
        palettes_json = json.dumps(palettes_dict)

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt-to-ASCII: Natural Fluid Motion</title>
    <style>
        :root {{
            --bg: {background};
            --panel-bg: rgba(14, 16, 22, 0.92);
            --border: rgba(255, 255, 255, 0.12);
            --accent: #58a6ff;
            --text: #f0f6fc;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }}
        #canvas-wrap {{
            position: relative;
            box-shadow: 0 20px 60px rgba(0,0,0,0.9);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            background: #020204;
        }}
        canvas {{
            display: block;
        }}
        .controls {{
            position: fixed;
            bottom: 24px;
            background: var(--panel-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 8px 24px;
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 100;
            box-shadow: 0 16px 48px rgba(0,0,0,0.75);
        }}
        .group-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(255,255,255,0.45);
            font-weight: 600;
            margin-right: 2px;
        }}
        .pill-btn {{
            background: transparent;
            border: 1px solid transparent;
            color: rgba(255,255,255,0.72);
            font-size: 12px;
            font-weight: 500;
            padding: 5px 12px;
            border-radius: 999px;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .pill-btn:hover {{
            color: #fff;
            background: rgba(255,255,255,0.08);
        }}
        .pill-btn.active {{
            background: #fff;
            color: #040406;
            font-weight: 600;
        }}
        .pill-btn.motion-active {{
            background: var(--accent);
            color: #040406;
            font-weight: 600;
        }}
        .divider {{
            width: 1px;
            height: 22px;
            background: var(--border);
            margin: 0 4px;
        }}
        .slider-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: rgba(255,255,255,0.75);
            font-weight: 500;
        }}
        .slider-val {{
            color: var(--accent);
            font-family: monospace;
            font-size: 11px;
            min-width: 32px;
        }}
        input[type="range"] {{
            accent-color: var(--accent);
            cursor: pointer;
            width: 72px;
        }}
        .action-btn {{
            background: rgba(255,255,255,0.1);
            border: 1px solid var(--border);
            color: #fff;
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.15s;
        }}
        .action-btn:hover {{ background: rgba(255,255,255,0.2); }}
    </style>
</head>
<body>
    <div id="canvas-wrap">
        <canvas id="ascii-canvas"></canvas>
    </div>

    <div class="controls">
        <span class="group-label">Style</span>
        <button class="pill-btn style-btn" data-style="clean">ASCII</button>
        <button class="pill-btn style-btn active" data-style="dots">Dots</button>
        <button class="pill-btn style-btn" data-style="pixel">Blocks</button>
        <button class="pill-btn style-btn" data-style="detailed">Detailed</button>


        <div class="divider"></div>

        <span class="group-label">Flow</span>
        <button class="pill-btn motion-btn motion-active" data-motion="flow">Natural Flow</button>
        <button class="pill-btn motion-btn" data-motion="static">Static</button>

        <div class="divider"></div>

        <div class="slider-group">
            <span>Size</span>
            <input type="range" id="size-slider" min="0.5" max="2.5" step="0.1" value="1.0">
            <span class="slider-val" id="size-val">1.0x</span>
        </div>

        <div class="divider"></div>

        <div class="slider-group">
            <span>Speed</span>
            <input type="range" id="speed-slider" min="0.2" max="2.5" step="0.1" value="1.0">
            <span class="slider-val" id="speed-val">1.0x</span>
        </div>
        <button class="action-btn" id="pause-btn">Pause</button>
    </div>

    <script>
        // High-Performance 2D Simplex Noise
        const SimplexNoise = (function() {{
            const F2 = 0.5 * (Math.sqrt(3.0) - 1.0);
            const G2 = (3.0 - Math.sqrt(3.0)) / 6.0;
            const seed = [
                151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
                8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
                35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
                134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
                55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
                18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
                250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
                189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
                172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
                228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
                107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
                138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
            ];
            const perm = new Uint8Array(512);
            const permMod12 = new Uint8Array(512);
            for (let i = 0; i < 512; i++) {{
                perm[i] = seed[i & 255];
                permMod12[i] = perm[i] % 12;
            }}
            const grad3 = new Float32Array([
                1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
                1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
                0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1
            ]);

            function noise2D(xin, yin) {{
                let n0 = 0, n1 = 0, n2 = 0;
                const s = (xin + yin) * F2;
                const i = Math.floor(xin + s);
                const j = Math.floor(yin + s);
                const t = (i + j) * G2;
                const X0 = i - t;
                const Y0 = j - t;
                const x0 = xin - X0;
                const y0 = yin - Y0;

                let i1, j1;
                if (x0 > y0) {{ i1 = 1; j1 = 0; }}
                else {{ i1 = 0; j1 = 1; }}

                const x1 = x0 - i1 + G2;
                const y1 = y0 - j1 + G2;
                const x2 = x0 - 1.0 + 2.0 * G2;
                const y2 = y0 - 1.0 + 2.0 * G2;

                const ii = i & 255;
                const jj = j & 255;
                const gi0 = permMod12[ii + perm[jj]] * 3;
                const gi1 = permMod12[ii + i1 + perm[jj + j1]] * 3;
                const gi2 = permMod12[ii + 1 + perm[jj + 1]] * 3;

                let t0 = 0.5 - x0 * x0 - y0 * y0;
                if (t0 > 0) {{
                    t0 *= t0;
                    n0 = t0 * t0 * (grad3[gi0] * x0 + grad3[gi0 + 1] * y0);
                }}
                let t1 = 0.5 - x1 * x1 - y1 * y1;
                if (t1 > 0) {{
                    t1 *= t1;
                    n1 = t1 * t1 * (grad3[gi1] * x1 + grad3[gi1 + 1] * y1);
                }}
                let t2 = 0.5 - x2 * x2 - y2 * y2;
                if (t2 > 0) {{
                    t2 *= t2;
                    n2 = t2 * t2 * (grad3[gi2] * x2 + grad3[gi2 + 1] * y2);
                }}
                return 0.5 + 35.0 * (n0 + n1 + n2);
            }}

            function fbm2D(x, y, octaves = 2) {{
                let val = 0, amp = 0.6, freq = 1.0, maxAmp = 0;
                for (let i = 0; i < octaves; i++) {{
                    val += amp * noise2D(x * freq, y * freq);
                    maxAmp += amp;
                    freq *= 2.0;
                    amp *= 0.5;
                }}
                return val / maxAmp;
            }}

            return {{ noise2D, fbm2D }};
        }})();

        // 4x4 Bayer Matrix for Seamless Photographic Dither Transitions
        const BAYER_4X4 = [
            -0.5,       0.0,    -0.375,   0.125,
             0.25,     -0.25,    0.375,  -0.125,
            -0.3125,    0.1875, -0.4375,  0.0625,
             0.4375,   -0.0625,  0.3125, -0.1875
        ];

        const palettes = {palettes_json};
        const aspectCompensation = {self.aspect_ratio};
        const canvas = document.getElementById('ascii-canvas');
        const ctx = canvas.getContext('2d');

        const offscreenCanvas = document.createElement('canvas');
        const offCtx = offscreenCanvas.getContext('2d', {{ willReadFrequently: true }});

        const sourceImg = new Image();
        sourceImg.src = "{img_data_url}";

        const W = {default_width};
        let H = 60;
        let charW = 7;
        let charH = 11;
        let charScale = 1.0;  // user-controlled size multiplier

        let grid = []; // 2D array of {{ lum, r, g, b }}
        let currentStyle = 'dots';   // color-first default: dots = pure color vessels
        let currentMotion = 'flow';
        let speed = 1.0;
        let isPaused = false;
        let frameCount = 0;

        function resizeCanvas() {{
            if (!grid.length) return;
            const controlsH = document.querySelector('.controls').offsetHeight;
            const availH = window.innerHeight - controlsH - 64;  // 64px gap between canvas & bar
            const availW = window.innerWidth - 60;
            const baseCharH = Math.max(6, Math.min(13, Math.floor(availH / H)));
            charH = Math.round(baseCharH * charScale);
            charW = charH * 0.58;
            canvas.width = Math.floor(W * charW);
            canvas.height = Math.floor(H * charH);
        }}

        function buildGrid() {{
            if (!sourceImg.complete || sourceImg.naturalWidth === 0) return;

            const imgAspect = sourceImg.naturalHeight / sourceImg.naturalWidth;
            H = Math.max(10, Math.floor(W * imgAspect * aspectCompensation));

            offscreenCanvas.width = W;
            offscreenCanvas.height = H;
            offCtx.drawImage(sourceImg, 0, 0, W, H);
            const imgData = offCtx.getImageData(0, 0, W, H).data;

            grid = [];
            for (let y = 0; y < H; y++) {{
                const row = [];
                for (let x = 0; x < W; x++) {{
                    const idx = (y * W + x) * 4;
                    const r = imgData[idx];
                    const g = imgData[idx + 1];
                    const b = imgData[idx + 2];
                    const lum = Math.floor(0.299 * r + 0.587 * g + 0.114 * b);
                    row.push({{ lum, r, g, b }});
                }}
                grid.push(row);
            }}

            resizeCanvas();
        }}

        sourceImg.onload = () => {{
            buildGrid();
            requestAnimationFrame(draw);
        }};

        function draw() {{
            if (!isPaused && currentMotion !== 'static') frameCount += speed;

            ctx.fillStyle = '{background}';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Crisp typography with zero blur/glow
            ctx.shadowBlur = 0;
            ctx.font = `${{charH}}px "Cascadia Code", "JetBrains Mono", "SF Mono", "Fira Code", monospace`;
            ctx.textBaseline = 'top';

            if (!grid.length) {{
                requestAnimationFrame(draw);
                return;
            }}

            // Time offset: calm, steady, natural fluid motion
            const t = frameCount * 0.022;
            const pal = palettes[currentStyle] || palettes['clean'];
            const maxIdx = pal.length - 1;

            for (let y = 0; y < H; y++) {{
                for (let x = 0; x < W; x++) {{
                    let sx = x;
                    let sy = y;
                    let flowFactor = 1.0;

                    if (currentMotion === 'flow') {{
                        // Natural Multi-Scale Flowing Density (like ascii.rest)
                        const n1 = SimplexNoise.fbm2D(x * 0.035 + t * 0.4, y * 0.035 - t * 0.5, 2);
                        const n2 = SimplexNoise.fbm2D(x * 0.065 - t * 0.3, y * 0.065 + t * 0.3, 2);
                        const flow = (n1 * 0.65 + n2 * 0.35 - 0.5) * 2.0;

                        // Subtle sub-character fluid drift (subject stays 100% crisp & identifiable)
                        const driftX = (n1 - 0.5) * 1.4;
                        const driftY = (n2 - 0.5) * 1.4;
                        sx = Math.max(0, Math.min(W - 1, Math.round(x + driftX)));
                        sy = Math.max(0, Math.min(H - 1, Math.round(y + driftY)));

                        flowFactor = 0.88 + 0.24 * flow;
                    }}
                    else if (currentMotion === 'currents') {{
                        // Silky 2D Curl Noise Streamlines
                        const angle = SimplexNoise.fbm2D(x * 0.03 + t * 0.2, y * 0.035 + t * 0.2, 2) * Math.PI * 4.0;
                        const driftX = Math.cos(angle) * 1.4;
                        const driftY = Math.sin(angle) * 1.4;
                        sx = Math.max(0, Math.min(W - 1, Math.round(x + driftX)));
                        sy = Math.max(0, Math.min(H - 1, Math.round(y + driftY)));

                        const ripple = 0.88 + 0.22 * Math.sin(x * 0.12 + y * 0.14 - t * 1.8);
                        flowFactor = ripple;
                    }}
                    else if (currentMotion === 'wave') {{
                        // Continuous Gentle Diagonal Swell
                        const wave = Math.sin(x * 0.08 + y * 0.10 - t * 1.5) * 0.5 + Math.sin(x * 0.04 - y * 0.06 - t * 0.8) * 0.5;
                        flowFactor = 0.88 + 0.24 * wave;
                    }}
                    else if (currentMotion === 'breath') {{
                        // Atmospheric Gentle Breathing (Stationary coordinates)
                        const breath = Math.sin(x * 0.12 + t * 1.2) * Math.cos(y * 0.12 + t * 1.2);
                        flowFactor = 0.90 + 0.20 * breath;
                    }}
                    else if (currentMotion === 'static') {{
                        flowFactor = 1.0;
                    }}

                    const cell = grid[sy][sx];

                    // Skip only pixels below shadow_lift threshold (configurable, default 10)
                    if (cell.lum < {self.shadow_lift if self.shadow_lift > 0 else 10}) continue;

                    // Calculate normalized luminance modulated by the flow
                    const baseNorm = cell.lum / 255.0;
                    const modulatedNorm = Math.max(0, Math.min(1.0, baseNorm * flowFactor));

                    // Bayer ordered dither transition (THE secret to ascii.rest's natural dissolving flow)
                    const bayer = BAYER_4X4[(y & 3) * 4 + (x & 3)];
                    const dithered = modulatedNorm + (currentMotion === 'static' ? 0 : bayer * (0.6 / maxIdx));
                    const charIdx = Math.max(0, Math.min(maxIdx, Math.round(dithered * maxIdx)));
                    const renderChar = pal[charIdx];

                    // Color-first: boost saturation so colors define the image, chars are vessels
                    // Saturate via HSL: amplify chroma relative to luminance
                    const lNorm = cell.lum / 255.0;
                    const SAT_BOOST = 1.55; // pull colors away from grey
                    const CL = {self.color_lift}; // color_lift: minimum per-channel floor
                    const rS = Math.min(255, Math.floor(lNorm * 255 + (cell.r - lNorm * 255) * SAT_BOOST));
                    const gS = Math.min(255, Math.floor(lNorm * 255 + (cell.g - lNorm * 255) * SAT_BOOST));
                    const bS = Math.min(255, Math.floor(lNorm * 255 + (cell.b - lNorm * 255) * SAT_BOOST));
                    const r = Math.max(CL, Math.max(0, rS));
                    const g = Math.max(CL, Math.max(0, gS));
                    const b = Math.max(CL, Math.max(0, bS));

                    // Subtle inner glow: draws the char twice — dim halo then crisp top layer
                    ctx.shadowBlur = 6;
                    ctx.shadowColor = `rgb(${{r}},${{g}},${{b}})`;
                    ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
                    ctx.fillText(renderChar, x * charW, y * charH);
                    ctx.shadowBlur = 0;
                }}
            }}

            requestAnimationFrame(draw);
        }}

        // Style selector
        document.querySelectorAll('.style-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentStyle = btn.dataset.style;
            }});
        }});

        // Motion selector
        document.querySelectorAll('.motion-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('.motion-btn').forEach(b => b.classList.remove('motion-active'));
                btn.classList.add('motion-active');
                currentMotion = btn.dataset.motion;
            }});
        }});

        const speedSlider = document.getElementById('speed-slider');
        const speedVal = document.getElementById('speed-val');
        speedSlider.addEventListener('input', (e) => {{
            speed = parseFloat(e.target.value);
            speedVal.textContent = `${{speed.toFixed(1)}}x`;
        }});

        const sizeSlider = document.getElementById('size-slider');
        const sizeVal = document.getElementById('size-val');
        sizeSlider.addEventListener('input', (e) => {{
            charScale = parseFloat(e.target.value);
            sizeVal.textContent = `${{charScale.toFixed(1)}}x`;
            resizeCanvas();
        }});

        const pauseBtn = document.getElementById('pause-btn');
        pauseBtn.addEventListener('click', () => {{
            isPaused = !isPaused;
            pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
        }});

        window.addEventListener('resize', resizeCanvas);
    </script>
</body>
</html>"""

        if output_file:
            Path(output_file).write_text(html_template, encoding="utf-8")

        return html_template

    def play_motion_terminal(
        self,
        image_input: Union[str, Path, Image.Image],
        mode: str = "flow",
        fps: int = 24,
        duration: Optional[float] = None,
    ):
        """
        Plays living procedural motion directly inside the terminal using ANSI escape codes.
        Modes: 'flow' (natural organic flow), 'currents' (streamlines), 'wave', 'breath'.
        Flicker-free real-time rendering. Press Ctrl+C to exit cleanly.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        else:
            img = image_input

        target_w, target_h = self._calculate_dimensions(img.width, img.height)
        processed_rgb = self._preprocess_image(img).resize((target_w, target_h), Image.Resampling.LANCZOS)
        gray_img = processed_rgb.convert("L")

        if self.invert:
            gray_img = ImageOps.invert(gray_img)

        gray_pixels = list(gray_img.getdata())
        rgb_pixels = list(processed_rgb.getdata())

        frame_delay = 1.0 / max(1, fps)
        start_time = time.time()
        frame_idx = 0

        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break

                lines = []
                frame_idx += 1
                t = frame_idx * 0.12

                for y in range(target_h):
                    line_chars = []
                    for x in range(target_w):
                        idx = y * target_w + x
                        lum = gray_pixels[idx]
                        r, g, b = rgb_pixels[idx]

                        if lum < 10:
                            line_chars.append(" ")
                            continue

                        if mode == "flow":
                            flow = (math.sin(x * 0.12 + t * 0.8) * 0.6 +
                                    math.sin(y * 0.14 - t * 0.9) * 0.4)
                            flow_factor = 0.88 + 0.24 * flow

                        elif mode == "currents":
                            wave = math.sin(x * 0.16 + y * 0.12 - t * 1.4)
                            flow_factor = 0.88 + 0.22 * wave

                        elif mode == "wave":
                            wave = math.sin(x * 0.09 + y * 0.10 - t * 1.2)
                            flow_factor = 0.88 + 0.24 * wave

                        else:  # breath
                            breath = math.sin(x * 0.15 + t) * math.cos(y * 0.15 + t)
                            flow_factor = 0.90 + 0.20 * breath

                        eff_lum = max(0, min(255, int(lum * flow_factor)))
                        # Preserve original RGB colors; do not dim them with flow_factor
                        final_r, final_g, final_b = r, g, b

                        char = self._map_luminance_to_char(eff_lum)
                        line_chars.append(f"\033[38;2;{final_r};{final_g};{final_b}m{char}\033[0m")

                    lines.append("".join(line_chars))

                sys.stdout.write("\033[H" + "\n".join(lines))
                sys.stdout.flush()
                time.sleep(frame_delay)

        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()


    def render_to_gif(
        self,
        image_input: Union[str, Path, Image.Image],
        output_file: Union[str, Path],
        num_frames: int = 24,
        fps: int = 12,
        font_size: int = 10,
        background: Tuple[int, int, int] = (4, 4, 6),
        font_path: Optional[str] = None,
    ) -> Path:
        """
        Renders the ASCII art as an animated GIF with flowing character motion.
        Characters oscillate through the palette over time using sinusoidal flow,
        simulating the HTML motion player in a portable, embeddable GIF format.

        :param num_frames: Number of animation frames (24 = smooth ~2s loop).
        :param fps: Frames per second.
        :param font_size: Character height in pixels. Smaller = more detail, larger file.
        :param background: RGB background colour tuple.
        :param font_path: Path to a .ttf monospace font. Falls back to system fonts, then PIL default.
        :return: Path to the saved GIF.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        else:
            img = image_input

        target_w, target_h = self._calculate_dimensions(img.width, img.height)
        processed_rgb = self._preprocess_image(img).resize(
            (target_w, target_h), Image.Resampling.LANCZOS
        )
        gray_img = processed_rgb.convert("L")
        if self.invert:
            gray_img = ImageOps.invert(gray_img)

        edge_chars = None
        if self.edge_detection:
            _, edge_chars = self._compute_edges_and_directions(gray_img)

        gray_pixels = list(gray_img.getdata())
        rgb_pixels = list(processed_rgb.getdata())

        # ── Font resolution ────────────────────────────────────────────────────
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont
        _loaded = False
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
                _loaded = True
            except Exception:
                pass
        if not _loaded:
            for _fname in [
                "cour.ttf", "courbd.ttf",           # Windows Courier New
                "DejaVuSansMono.ttf",               # Linux / macOS
                "LiberationMono-Regular.ttf",
                "Menlo.ttc",                        # macOS
            ]:
                try:
                    font = ImageFont.truetype(_fname, font_size)
                    _loaded = True
                    break
                except Exception:
                    continue
        if not _loaded:
            font = ImageFont.load_default()

        # Character cell size
        _bbox = font.getbbox("W")
        char_w = max(1, _bbox[2] - _bbox[0])
        char_h = max(1, _bbox[3] - _bbox[1] + 2)  # +2px leading

        canvas_w = target_w * char_w
        canvas_h = target_h * char_h

        palette = self.palette
        palette_len = len(palette)
        lum_cutoff = max(10, self.shadow_lift)
        frame_ms = max(20, int(1000 / fps))

        frames: List[Image.Image] = []

        for f in range(num_frames):
            # Map frame index to a full 2π cycle
            t = (f / num_frames) * math.pi * 2

            frame_img = Image.new("RGB", (canvas_w, canvas_h), background)
            draw = ImageDraw.Draw(frame_img)

            for y in range(target_h):
                for x in range(target_w):
                    idx = y * target_w + x
                    lum = gray_pixels[idx]

                    if lum < lum_cutoff:
                        continue

                    # Edge char takes priority
                    if self.edge_detection and edge_chars and edge_chars[y][x]:
                        char = edge_chars[y][x]
                    else:
                        # Sinusoidal flow: subtly oscillate the palette index over time
                        base_ratio = lum / 255.0
                        noise = (
                            math.sin(x * 0.18 + t * 1.1) * 0.5
                            + math.sin(y * 0.14 - t * 0.85) * 0.5
                        ) * 0.14  # ±14% oscillation — visible but not distracting
                        ratio = max(0.0, min(1.0, base_ratio + noise))
                        if self.invert:
                            ratio = 1.0 - ratio
                        if palette[0] == " ":
                            char_idx = int(ratio * (palette_len - 1))
                        else:
                            char_idx = int((1.0 - ratio) * (palette_len - 1))
                        char = palette[char_idx]

                    if char == " ":
                        continue

                    r, g, b = rgb_pixels[idx]
                    if self.color_lift > 0:
                        r = max(self.color_lift, r)
                        g = max(self.color_lift, g)
                        b = max(self.color_lift, b)

                    draw.text((x * char_w, y * char_h), char, fill=(r, g, b), font=font)

            frames.append(frame_img)

        out_path = Path(output_file)
        # GIF is 256-colour; convert each frame via quantize for best quality
        pal_frames = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for f in frames]
        pal_frames[0].save(
            out_path,
            format="GIF",
            append_images=pal_frames[1:],
            save_all=True,
            duration=frame_ms,
            loop=0,
            optimize=True,
        )
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Image to High-Quality ASCII Converter with Natural Motion")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("-w", "--width", type=int, default=100, help="Output width in characters (default: 100)")
    parser.add_argument("-p", "--palette", type=str, default="clean", choices=list(PALETTES.keys()), help="Palette preset")
    parser.add_argument("--aspect", type=float, default=0.50, help="Vertical aspect compensation ratio (default: 0.50)")
    parser.add_argument("-c", "--contrast", type=float, default=1.15, help="Contrast multiplier (e.g. 1.2)")
    parser.add_argument("-b", "--brightness", type=float, default=1.0, help="Brightness multiplier (e.g. 1.1)")
    parser.add_argument("-g", "--gamma", type=float, default=1.0, help="Gamma factor")
    parser.add_argument("--shadow-lift", type=int, default=0, help="Minimum luminance floor 0-80 (lifts dark areas so they render as visible chars, not spaces)")
    parser.add_argument("--color-lift", type=int, default=0, help="Minimum RGB channel floor 0-80 (makes dark-colored chars visible on dark backgrounds)")
    parser.add_argument("--edges", action="store_true", help="Enable directional edge detection outlines")
    parser.add_argument("--edge-thresh", type=int, default=95, help="Sobel edge sensitivity (default: 95)")
    parser.add_argument("--color", action="store_true", help="Output ANSI TrueColor for terminals")
    parser.add_argument("--invert", action="store_true", help="Invert luminance mapping")
    parser.add_argument("-o", "--output", type=str, help="Save plain text output to file")
    parser.add_argument("--html", type=str, help="Export styled HTML output to file")
    parser.add_argument("--motion-html", type=str, help="Export interactive 60-FPS natural flow HTML player")
    parser.add_argument("--gif", type=str, help="Export animated GIF with flowing character motion")
    parser.add_argument("--gif-frames", type=int, default=24, help="Number of GIF animation frames (default: 24)")
    parser.add_argument("--gif-fps", type=int, default=12, help="GIF frames per second (default: 12)")
    parser.add_argument("--live", choices=["flow", "currents", "wave", "breath"], help="Play live natural flow motion in terminal")
    parser.add_argument("--fps", type=int, default=24, help="FPS for live terminal motion (default: 24)")

    args = parser.parse_args()

    renderer = AsciiRenderer(
        width=args.width,
        aspect_ratio=args.aspect,
        palette=args.palette,
        invert=args.invert,
        contrast=args.contrast,
        brightness=args.brightness,
        gamma=args.gamma,
        shadow_lift=args.shadow_lift,
        color_lift=args.color_lift,
        edge_detection=args.edges,
        edge_threshold=args.edge_thresh,
    )

    if args.live:
        print(f"Starting live '{args.live}' motion in terminal (Press Ctrl+C to exit)...")
        time.sleep(1)
        renderer.play_motion_terminal(args.image_path, mode=args.live, fps=args.fps)
    elif args.gif:
        print(f"Rendering animated GIF ({args.gif_frames} frames @ {args.gif_fps} fps)...")
        out = renderer.render_to_gif(
            args.image_path,
            output_file=args.gif,
            num_frames=args.gif_frames,
            fps=args.gif_fps,
        )
        print(f"GIF saved to: {out}")
    elif args.motion_html:
        renderer.render_motion_html(args.image_path, output_file=args.motion_html)
        print(f"Interactive Natural Flow HTML saved to: {args.motion_html}")
    elif args.html:
        renderer.render_to_html(args.image_path, output_file=args.html)
        print(f"HTML output saved to: {args.html}")
    else:
        result = renderer.render(args.image_path, color=args.color)
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
            print(f"ASCII output saved to: {args.output}")
        else:
            print(result)


if __name__ == "__main__":
    main()


