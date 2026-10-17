"""
main.py - Test Runner for ASCII Rendering Engine
================================================
Runs tests on images in test_images/ to evaluate:
- Baseline luminance mapping (standard vs detailed vs blocks palettes)
- Edge-preserving directional outlines (Phase 2 feature)
- Contrast adjustments
- Output to terminal and saved files (.txt and .html)
"""

import sys
from pathlib import Path
from customRenderer import AsciiRenderer, PALETTES

OUTPUT_DIR = Path("output")
TEST_IMAGES_DIR = Path("test_images")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def get_available_images():
    """Finds all image files in test_images/ directory or project root."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = []

    if TEST_IMAGES_DIR.exists():
        images.extend([p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in valid_exts])

    if not images:
        images.extend([p for p in Path(".").iterdir() if p.suffix.lower() in valid_exts])

    return sorted(images)


def test_single_image(image_path: Path):
    """
    Renders an image under different configurations and saves the outputs.
    """
    ensure_output_dir()
    stem = image_path.stem
    print(f"\n{'=' * 60}")
    print(f"Testing on: {image_path.name}")
    print(f"{'=' * 60}\n")

    # 1. Clean Studio ASCII (10-ramp, crisp, zero noise)
    print("[1/6] Running Clean Studio ASCII...")
    clean_renderer = AsciiRenderer(
        width=130,
        aspect_ratio=0.50,
        palette="clean",
        contrast=1.4,
        saturation=1.2,
        gamma=1.2,
        edge_detection=False,
    )
    clean_ascii = clean_renderer.render(image_path)
    # (OUTPUT_DIR / f"{stem}_clean_ascii.txt").write_text(clean_ascii, encoding="utf-8")  # Disabled per user request
    print(clean_ascii[:500] + "\n... (truncated, full saved to output/)\n")

    # 2. Smooth Detailed Typography (70-ramp)
    print("[2/6] Running Smooth Detailed Typography...")
    detailed_renderer = AsciiRenderer(
        width=130,
        aspect_ratio=0.50,
        palette="detailed",
        contrast=1.4,
        saturation=1.2,
        gamma=1.2,
        edge_detection=False,
    )
    detailed_ascii = detailed_renderer.render(image_path)
    # (OUTPUT_DIR / f"{stem}_detailed.txt").write_text(detailed_ascii, encoding="utf-8")  # Disabled per user request

    # 3. Unicode Block Elements (Pixel Art Style)
    print("[3/6] Running Unicode Block Elements...")
    blocks_renderer = AsciiRenderer(
        width=130,
        aspect_ratio=0.50,
        palette="blocks",
        contrast=1.2,
        edge_detection=False,
    )
    blocks_ascii = blocks_renderer.render(image_path)
    # (OUTPUT_DIR / f"{stem}_blocks.txt").write_text(blocks_ascii, encoding="utf-8")  # Disabled per user request

    # 4. Ultra-Sharp Unicode Braille (2x4 subpixels, razor clean)
    print("[4/6] Running High-Resolution Unicode Braille...")
    braille_ascii = clean_renderer.render_braille(image_path, threshold=110)
    # (OUTPUT_DIR / f"{stem}_braille.txt").write_text(braille_ascii, encoding="utf-8")  # Disabled per user request

    # 5. Clean Full-Color Studio HTML (No slash clutter, zero glow)
    print("[5/6] Exporting Clean Studio Full-Color HTML...")
    html_output_path = OUTPUT_DIR / f"{stem}_color.html"
    clean_renderer.render_to_html(image_path, output_file=html_output_path, font_size=9)
    print(f"Saved: {html_output_path}")

    # 6. Interactive 60-FPS Natural Fluid Motion Player (Volcano Flow & Streamlines)
    print("[6/6] Exporting 60-FPS Natural Fluid Motion Player (Volcano Flow)...")
    motion_html_path = OUTPUT_DIR / f"{stem}_motion.html"
    # Use static mode so the motion HTML shows the same ASCII characters as the static HTML
    clean_renderer.render_motion_html(image_path, output_file=motion_html_path, font_size=10, motion='static')
    print(f"Saved: {motion_html_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Runner for Prompt-to-ASCII with Natural Fluid Motion")
    parser.add_argument("--live", choices=["volcano", "liquid", "wind", "shimmer"], help="Play live organic motion in terminal")
    parser.add_argument("--image", type=str, help="Specific image path to test")
    args = parser.parse_args()

    images = get_available_images()
    if not images:
        print("No images found in 'test_images/' or project root.")
        print("Supported formats: .jpg, .jpeg, .png, .webp, .bmp")
        sys.exit(1)

    selected_img = Path(args.image) if args.image else images[0]

    if args.live:
        print(f"\nPlaying live '{args.live}' motion for {selected_img.name} in terminal...")
        print("Press Ctrl+C at any time to exit.\n")
        import time
        time.sleep(1)
        renderer = AsciiRenderer(width=90, aspect_ratio=0.50, palette="clean", contrast=1.2, edge_detection=False)
        renderer.play_motion_terminal(selected_img, mode=args.live, fps=24)
        return

    print(f"Found {len(images)} test image(s):")
    for img in images:
        print(f" - {img}")

    # Run tests on all discovered images
    for img in images:
        test_single_image(img)

    print(f"\nAll tests completed! Check the '{OUTPUT_DIR.resolve()}' folder for results.")

if __name__ == "__main__":
    main()

