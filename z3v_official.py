#!/usr/bin/env python3
"""
ZetaLvX custom script (batch/single) for Hunyuan3D-2.1 shape + paint.

- Author repo: ZetaLvX (GitHub)
- Based on: the official Hunyuan3D (Hy3D) demo code from the Tencent/Hunyuan3D-2.1 GitHub project.
- This is a revised/extended version for my personal workflow (batch processing, optional background removal,
  headless operation without Blender, and optional single-file mode).
"""

import os
import sys
import types
import argparse
from pathlib import Path

# -----------------------------
# SETTINGS YOU MAY WANT TO EDIT
# -----------------------------
MODEL_PATH = "/path/to/hy3d/models/Hunyuan3D-2.1"  # folder containing hunyuan3d-dit-v2-1 etc.
MAX_NUM_VIEW = 8        # typical range: 6..12 (higher => more VRAM)
RESOLUTION = 512        # typical: 512 or 768 (higher => more VRAM)

# -----------------------------
# SAFE "BYPASS" FOR bpy IMPORTS
# -----------------------------
# Some upstream modules import bpy (Blender). We stub it to avoid crashes / dependency.
if "bpy" not in sys.modules:
    sys.modules["bpy"] = types.ModuleType("bpy")

SCRIPT_DIR = Path(__file__).resolve().parent
HY3DSHAPE_DIR = (SCRIPT_DIR / "hy3dshape").resolve()
HY3DPAINT_DIR = (SCRIPT_DIR / "hy3dpaint").resolve()

# Add project modules to sys.path (so imports work without installing as packages)
sys.path.insert(0, str(HY3DSHAPE_DIR))
sys.path.insert(0, str(HY3DPAINT_DIR))

from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# Optional torchvision compat patch (kept safe with try/except)
try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("[WARN] torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"[WARN] Failed to apply torchvision fix: {e}")

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

# Optional: patch convert_obj_to_glb to avoid Blender dependency (kept for safety)
try:
    from DifferentiableRenderer import mesh_utils
    import trimesh

    def _convert_obj_to_glb_headless(obj_path, glb_path):
        mesh = trimesh.load(obj_path, force="mesh")
        mesh.export(glb_path)
        return glb_path

    mesh_utils.convert_obj_to_glb = _convert_obj_to_glb_headless
    print("[OK] Patched convert_obj_to_glb to trimesh (no bpy).")
except Exception as e:
    print(f"[WARN] Could not patch convert_obj_to_glb (may still be ok): {e}")


def is_valid_glb(path: Path) -> bool:
    """Quick signature check for GLB (starts with b'glTF')."""
    try:
        with path.open("rb") as f:
            return f.read(4) == b"glTF"
    except Exception:
        return False


def looks_like_obj_text(path: Path) -> bool:
    """Heuristic: detect OBJ text saved with the wrong extension."""
    try:
        with path.open("rb") as f:
            head = f.read(64)
        return (
            (b"mtllib" in head)
            or head.lstrip().startswith(b"#")
            or head.lstrip().startswith(b"o ")
            or head.lstrip().startswith(b"v ")
        )
    except Exception:
        return False


def has_alpha_png(p: Path) -> bool:
    """True if PNG likely contains alpha channel."""
    if p.suffix.lower() != ".png":
        return False
    try:
        im = Image.open(p)
        if im.mode in ("RGBA", "LA"):
            return True
        # Some PNGs have alpha in palette mode
        return ("transparency" in im.info)
    except Exception:
        return False


def remove_background_to_png(src: Path, dst_png: Path, force: bool = False) -> Path:
    """
    Create an RGBA PNG with background removed using U^2-Net (via BackgroundRemover).
    If src is already a PNG with alpha and not force -> returns src.
    Otherwise saves dst_png and returns dst_png.
    """
    if (not force) and has_alpha_png(src):
        return src

    dst_png.parent.mkdir(parents=True, exist_ok=True)

    im = Image.open(src)
    # Rembg path in hy3dshape expects RGB and outputs RGBA
    im_rgb = im.convert("RGB") if im.mode != "RGB" else im

    rembg = BackgroundRemover()
    out = rembg(im_rgb)  # returns PIL image with alpha
    out = out.convert("RGBA")
    out.save(dst_png)
    return dst_png


def collect_images(input_dir: Path) -> list[Path]:
    """Collect supported image files from input_dir (non-recursive)."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    return [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]


def main():
    parser = argparse.ArgumentParser(
        description="ZetaLvX: Batch/single Hunyuan3D-2.1 shape + paint on images."
    )

    # English flags (with Italian alias for backward compatibility)
    parser.add_argument(
        "--remove-bg",
        "--rembg",
        "--rimuovi-sfondo",
        action="store_true",
        help="Remove background for each image (U^2-Net) and use the resulting RGBA PNG as input."
    )
    parser.add_argument(
        "--force-remove",
        "--forza-rimozione",
        action="store_true",
        help="Force background removal even if the input is already a PNG with alpha."
    )

    # Single file mode
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Process a single image file instead of scanning ./input/. Example: --path /abs/or/rel/file.png"
    )

    args = parser.parse_args()

    input_dir = (SCRIPT_DIR / "input").resolve()
    output_dir = (SCRIPT_DIR / "output").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decide whether to process a single file or the whole input folder
    if args.path:
        single = Path(args.path).expanduser()
        if not single.is_absolute():
            single = (Path.cwd() / single).resolve()
        if not single.exists() or not single.is_file():
            print(f"[ERROR] --path file not found: {single}")
            return
        images = [single]
    else:
        images = collect_images(input_dir)

    if not images:
        print(f"[INFO] No images found in: {input_dir}")
        print("[INFO] Put one or more images into ./input and run again, or use --path <file>.")
        return

    # Load shape pipeline once (reused for all images)
    print(f"[INFO] Loading shape model from: {MODEL_PATH}")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_PATH)

    # Configure paint pipeline once (reused for all images)
    conf = Hunyuan3DPaintConfig(MAX_NUM_VIEW, RESOLUTION)
    conf.realesrgan_ckpt_path = str((HY3DPAINT_DIR / "ckpt" / "RealESRGAN_x4plus.pth").resolve())
    conf.multiview_cfg_path = str((HY3DPAINT_DIR / "cfgs" / "hunyuan-paint-pbr.yaml").resolve())
    conf.custom_pipeline = str((HY3DPAINT_DIR / "hunyuanpaintpbr").resolve())

    print("[INFO] Loading paint pipeline...")
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    for img_path in images:
        stem = img_path.stem
        per_item_dir = (output_dir / stem).resolve()
        per_item_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"[INFO] Processing: {img_path.name}")
        print(f"[INFO] Output dir:  {per_item_dir}")

        old_cwd = Path.cwd()
        os.chdir(per_item_dir)

        try:
            # Optionally remove background and generate a transparent PNG in the output folder
            input_for_model = img_path
            if args.remove_bg:
                bg_png = per_item_dir / f"{stem}_bgremoved.png"
                input_for_model = remove_background_to_png(
                    img_path, bg_png, force=args.force_remove
                )
                print(f"[OK] BG removed -> {Path(input_for_model).name}")

            # Load image for shapegen
            image = Image.open(input_for_model)

            # Ensure RGBA (shape + paint behave best with consistent RGBA input)
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            # SHAPE
            shape_glb = per_item_dir / f"{stem}_shape.glb"
            mesh = pipeline_shapegen(image=image)[0]
            mesh.export(str(shape_glb))
            print(f"[OK] Shape exported: {shape_glb.name}")

            # PAINT
            out_glb_maybe = per_item_dir / f"{stem}_textured.glb"
            paint_pipeline(
                mesh_path=str(shape_glb),
                image_path=str(input_for_model),
                output_mesh_path=str(out_glb_maybe),
            )

            # Fix wrong extension: if it's OBJ text saved as .glb, rename to .obj
            if out_glb_maybe.exists():
                if is_valid_glb(out_glb_maybe):
                    print(f"[OK] Textured GLB exported: {out_glb_maybe.name}")
                elif looks_like_obj_text(out_glb_maybe):
                    out_obj = per_item_dir / f"{stem}_textured.obj"
                    out_glb_maybe.replace(out_obj)
                    print(f"[FIX] Output was OBJ text, renamed to: {out_obj.name}")
                else:
                    print(f"[WARN] Output exists but is not valid GLB and doesn't look like OBJ: {out_glb_maybe.name}")
            else:
                print("[WARN] output_mesh_path not found after paint. Check logs in this folder.")

        except Exception as e:
            print(f"[ERROR] Failed on {img_path.name}: {e}")
        finally:
            os.chdir(old_cwd)

    print("\n[DONE] Completed.")
    print(f"[INFO] Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
