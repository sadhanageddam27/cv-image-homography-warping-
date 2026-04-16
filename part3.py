# part3_digits_warping.py  —  FINAL version (labels centered on clicks)
# Warp one digit image to another (e.g., 3-a → 3-b, 7-a → 7-b)
# using both Piecewise Affine (Triangular Mesh) and Thin-Plate Spline (TPS)
# and display live point numbering while selecting correspondences.

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial import Delaunay
from skimage.transform import PiecewiseAffineTransform, ThinPlateSplineTransform, warp
from skimage import img_as_float32, img_as_ubyte
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- helpers -----------------------------

def load_image(path: str):
    from imageio.v3 import imread
    img = imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img_as_float32(img / (np.max(img) if np.max(img) > 0 else 1.0))
    return img

def click_points_with_labels(img, title, k):
    """Interactive point selection with live numbering (P1, P2, …) centered on the dot."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    pts = []

    def onclick(event):
        if event.inaxes != ax or len(pts) >= k:
            return
        x, y = event.xdata, event.ydata
        pts.append([x, y])
        # red dot
        ax.plot(x, y, 'ro', markersize=5)
        # centered label at the dot
        ax.text(x, y, f"P{len(pts)}",
                color='yellow', fontsize=10,
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.2',
                          facecolor='black', alpha=0.6, edgecolor='none'))
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    if len(pts) != k:
        raise RuntimeError(f"Expected {k} clicks, got {len(pts)}.")
    return np.array(pts, dtype=np.float64)

def save_labeled_points(img, pts, out_path, title, color='lime'):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.scatter(pts[:,0], pts[:,1], s=50, c=color, edgecolors='black', linewidths=0.8)
    for i,(x,y) in enumerate(pts, start=1):
        ax.text(x, y, f"P{i}", fontsize=9, color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'))
    ax.set_title(title); ax.axis('off'); fig.tight_layout()
    fig.savefig(out_path, dpi=220); plt.close(fig)

def save_mesh_overlay(img, pts, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.scatter(pts[:,0], pts[:,1], s=40, c='r', edgecolors='white', linewidths=1)
    try:
        tri = Delaunay(pts)
        for simplex in tri.simplices:
            p = pts[simplex]
            ax.plot([p[0,0], p[1,0]], [p[0,1], p[1,1]], 'y-', lw=1)
            ax.plot([p[1,0], p[2,0]], [p[1,1], p[2,1]], 'y-', lw=1)
            ax.plot([p[2,0], p[0,0]], [p[2,1], p[0,1]], 'y-', lw=1)
    except Exception:
        pass
    ax.set_title("Delaunay mesh (target frame)"); ax.axis('off'); fig.tight_layout()
    fig.savefig(out_path, dpi=220); plt.close(fig)

def boundary_points(W, H):
    return np.array([
        [0,0], [W-1,0], [0,H-1], [W-1,H-1],
        [W//2,0], [W//2,H-1], [0,H//2], [W-1,H//2]
    ], dtype=np.float64)

def ensure_outdir(name: str) -> Path:
    out = Path.cwd() / f"{name}_digits_part3"
    out.mkdir(parents=True, exist_ok=True)
    return out

# ------------------------------ main ------------------------------

def main():
    print("="*72)
    print("Part 3 (Updated): Digit Warping — Triangular Mesh vs Thin-Plate Spline")
    print("="*72)
    src_path = input("Source image path (e.g., 3-a.png): ").strip().strip('"')
    tgt_path = input("Target image path (e.g., 3-b.png): ").strip().strip('"')

    src = load_image(src_path)
    tgt = load_image(tgt_path)
    Hs, Ws = src.shape[:2]
    Ht, Wt = tgt.shape[:2]
    print(f"Loaded: source {Ws}x{Hs}, target {Wt}x{Ht}")

    try:
        k = int(input("How many corresponding points? (default 12): ").strip() or "12")
    except ValueError:
        k = 12
    k = max(6, k)

    # Click target points (with live numbering)
    print("\nStep 1) Click the", k, "TARGET points (order arbitrary, will match to source).")
    tgt_pts = click_points_with_labels(tgt, f"Click {k} TARGET points (P1..Pk)", k)

    # Click source points (with live numbering)
    print("\nStep 2) Click the", k, "corresponding SOURCE points in the SAME order.")
    src_pts = click_points_with_labels(src, f"Click {k} SOURCE points (same order)", k)

    # prepare output
    out_dir = ensure_outdir(Path(tgt_path).stem + "_from_" + Path(src_path).stem)

    save_labeled_points(tgt, tgt_pts, out_dir / "target_points_labeled.png",
                        "Target points (P1..Pk)", color='orange')
    save_labeled_points(src, src_pts, out_dir / "source_points_labeled.png",
                        "Source points (P1..Pk)", color='lime')

    tgt_b = boundary_points(Wt, Ht)
    src_b = boundary_points(Ws, Hs)
    tgt_aug = np.vstack([tgt_pts, tgt_b])
    src_aug = np.vstack([src_pts, src_b])

    save_mesh_overlay(tgt, tgt_aug, out_dir / "target_mesh.png")

    # --- Piecewise Affine ---
    print("\nWarping with Triangular Mesh (Piecewise Affine)...")
    t0 = time.time()
    pwa = PiecewiseAffineTransform()
    pwa.estimate(src_aug, tgt_aug)
    warped_pwa = warp(src, pwa, output_shape=(Ht, Wt), order=1, preserve_range=True)
    t_pwa = (time.time() - t0)*1000

    # --- Thin Plate Spline ---
    print("Warping with Thin-Plate Spline (TPS)...")
    t1 = time.time()
    tps = ThinPlateSplineTransform()
    tps.estimate(src_aug, tgt_aug)
    warped_tps = warp(src, tps, output_shape=(Ht, Wt), order=1, preserve_range=True)
    t_tps = (time.time() - t1)*1000

    src8 = img_as_ubyte(np.clip(src, 0, 1))
    tgt8 = img_as_ubyte(np.clip(tgt, 0, 1))
    pwa8 = img_as_ubyte(np.clip(warped_pwa, 0, 1))
    tps8 = img_as_ubyte(np.clip(warped_tps, 0, 1))

    Image.fromarray(src8).save(out_dir / "source.jpg")
    Image.fromarray(tgt8).save(out_dir / "target.jpg")
    Image.fromarray(pwa8).save(out_dir / "warped_piecewise_affine.jpg")
    Image.fromarray(tps8).save(out_dir / "warped_tps.jpg")

    # side-by-side
    fig = plt.figure(figsize=(16,5.5))
    for i,(im,title) in enumerate(
        [(src8,"Source"),
         (pwa8,"Warped (Mesh)"),
         (tps8,"Warped (TPS)"),
         (tgt8,"Target")], start=1):
        ax = fig.add_subplot(1,4,i)
        ax.imshow(im); ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    plt.show(block=True)
    fig.savefig(out_dir / "comparison_side_by_side.jpg", dpi=220)
    plt.close(fig)

    with open(out_dir / "readme_notes.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Digit warping {Path(src_path).name} -> {Path(tgt_path).name}\n"
            f"Piecewise Affine time: {t_pwa:.2f} ms\n"
            f"TPS time:              {t_tps:.2f} ms\n\n"
            "Observations:\n"
            "- Mesh: piecewise-linear, local; may have small seams at triangle borders.\n"
            "- TPS: smooth, globally continuous; better at preserving curvature.\n"
        )

    print("\n✓ Saved results to:", out_dir.resolve())
    print(f"Timing — Mesh: {t_pwa:.2f} ms | TPS: {t_tps:.2f} ms")
    print("="*72)

if __name__ == "__main__":
    main()
