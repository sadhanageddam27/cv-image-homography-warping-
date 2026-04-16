# homography_part2.py
# Part 2: Creative Applications of Homography (Scanner + AR)
# - Reuses the same Normalized-DLT approach/style as Part 1
# - Simple UI, zero clutter, exactly THREE outputs per demo

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path

# ============================== Part 1: minimal reuse ==============================

def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def ginput_points(img: np.ndarray, title: str, n: int = 4) -> np.ndarray:
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(title)
    pts = plt.ginput(n, timeout=0)
    plt.close()
    if len(pts) != n:
        raise RuntimeError(f"Expected {n} clicks, got {len(pts)}.")
    return np.array(pts, dtype=np.float64)

def _normalize_points(pts: np.ndarray):
    c = pts.mean(axis=0)
    shifted = pts - c
    md = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    s = np.sqrt(2) / (md + 1e-12)
    T = np.array([[s, 0, -s*c[0]],
                  [0, s, -s*c[1]],
                  [0, 0,       1]])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_n = (T @ pts_h.T).T[:, :2]
    return T, pts_n

def estimate_homography_normalized(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    assert src_pts.shape == dst_pts.shape and src_pts.shape[0] >= 4
    T1, src_n = _normalize_points(src_pts)
    T2, dst_n = _normalize_points(dst_pts)
    A = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        A.append([-x, -y, -1,  0,  0,  0, u*x, u*y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    H /= H[2, 2]
    return H

def ensure_outdir(name: str) -> Path:
    out = Path.cwd() / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving into: {out.resolve()}")
    return out

def draw_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    vis = img.copy()
    q = quad.astype(int)
    for i in range(4):
        p1 = tuple(q[i]); p2 = tuple(q[(i+1) % 4])
        cv2.line(vis, p1, p2, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(vis, p1, 6, (0, 255, 0), -1, cv2.LINE_AA)
    return vis

def save_panel_base_vs_ar(base: np.ndarray, comp: np.ndarray, out_path: Path):
    H = max(base.shape[0], comp.shape[0])

    def resize_h(img):
        h, w = img.shape[:2]
        if h == H: return img
        return cv2.resize(img, (int(round(w * (H / h))), H), interpolation=cv2.INTER_AREA)

    L = resize_h(base); R = resize_h(comp)
    pad, title = 32, 60
    W = L.shape[1] + R.shape[1] + pad * 3
    canvas = Image.new("RGB", (W, H + title + pad * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # left
    canvas.paste(Image.fromarray(L), (pad, pad + title))
    lt = "Base"
    tb = draw.textbbox((0, 0), lt)
    draw.text((pad + L.shape[1] // 2 - tb[2] // 2, pad), lt, fill=(0, 0, 0))
    # right
    xR = pad * 2 + L.shape[1]
    canvas.paste(Image.fromarray(R), (xR, pad + title))
    rt = "AR Composite"
    tb2 = draw.textbbox((0, 0), rt)
    draw.text((xR + R.shape[1] // 2 - tb2[2] // 2, pad), rt, fill=(0, 0, 0))

    canvas.save(out_path)

# ============================== Demo A: Document Scanner ==============================

def document_scanner():
    print("\n=== Part 2.A — Document Scanner ===")
    img_path = input("Enter document photo path: ").strip().strip('"')
    img = load_image_rgb(img_path)

    print("Click TL, TR, BR, BL on the document and close.")
    src = ginput_points(img, "Click TL, TR, BR, BL on the document (then close)")

    # auto rectangle size from edge lengths
    top  = np.linalg.norm(src[1] - src[0])
    left = np.linalg.norm(src[3] - src[0])
    aspect = (top / (left + 1e-9)) if left > 0 else 1.414
    out_w = 1000
    out_h = int(round(out_w / aspect))

    dst = np.array([[0, 0],
                    [out_w - 1, 0],
                    [out_w - 1, out_h - 1],
                    [0, out_h - 1]], dtype=np.float64)

    H = estimate_homography_normalized(src, dst)
    rectified = cv2.warpPerspective(img, H.astype(np.float64), (out_w, out_h), flags=cv2.INTER_LINEAR)

    # “Scanned” look
    gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
    scanned = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 25, 10)

    out_dir = ensure_outdir(Path(img_path).stem + "_scan")
    Image.fromarray(draw_quad(img, src)).save(out_dir / "points.jpg")          # (1) points
    Image.fromarray(rectified).save(out_dir / "rectified.jpg")                 # (2) rectified
    Image.fromarray(scanned).save(out_dir / "scanned_binary.png")              # (3) scanned

    print("✓ Saved: points.jpg, rectified.jpg, scanned_binary.png")

# ============================== Demo B: AR Poster/Logo ==============================

def load_texture_rgba(path: str) -> np.ndarray:
    tex = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if tex is None:
        raise FileNotFoundError(f"Cannot read texture: {path}")
    if tex.ndim == 2:
        tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGRA)
    elif tex.shape[2] == 3:
        # If no alpha, make it opaque so it ALWAYS shows
        tex = np.dstack([tex, np.full(tex.shape[:2], 255, np.uint8)])
    tex[..., :3] = cv2.cvtColor(tex[..., :3], cv2.COLOR_BGR2RGB)
    return tex  # RGBA

def overlay_texture_simple(base: np.ndarray, texture_rgba: np.ndarray, dst_quad: np.ndarray) -> np.ndarray:
    """
    Minimal & reliable: stretch fit (obvious), strict clipping, opaque alpha by default.
    """
    H_base, W_base = base.shape[:2]
    th, tw = texture_rgba.shape[:2]
    src_quad = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float64)

    H = estimate_homography_normalized(src_quad, dst_quad)

    tex_rgb = texture_rgba[..., :3]
    tex_a   = texture_rgba[..., 3]  # already opaque if no alpha in file

    warped_rgb = cv2.warpPerspective(tex_rgb, H, (W_base, H_base), flags=cv2.INTER_LINEAR)
    warped_a   = cv2.warpPerspective(tex_a,   H, (W_base, H_base), flags=cv2.INTER_LINEAR)

    # clip strictly to the quad
    clip = np.zeros((H_base, W_base), np.uint8)
    cv2.fillConvexPoly(clip, dst_quad.astype(np.int32), 255)
    warped_a = cv2.bitwise_and(warped_a, clip)

    a = (warped_a.astype(np.float32) / 255.0)[..., None]
    comp = (warped_rgb.astype(np.float32) * a + base.astype(np.float32) * (1.0 - a)).astype(np.uint8)
    return comp

def ar_insert():
    print("\n=== Part 2.B — AR Effect (Poster/Logo on a Plane) ===")
    base_path = input("Enter base photo path (scene): ").strip().strip('"')
    tex_path  = input("Enter texture image path (poster/logo): ").strip().strip('"')

    base = load_image_rgb(base_path)
    texture = load_texture_rgba(tex_path)

    print("Click TL, TR, BR, BL on the TARGET SURFACE (e.g., book cover) and close.")
    dst = ginput_points(base, "Click TL, TR, BR, BL on the target surface (then close)")

    comp = overlay_texture_simple(base, texture, dst)

    out_dir = ensure_outdir(Path(base_path).stem + "_AR")
    Image.fromarray(draw_quad(base, dst)).save(out_dir / "points.jpg")            # (1)
    Image.fromarray(comp).save(out_dir / "ar_composite.jpg")                      # (2)
    save_panel_base_vs_ar(base, comp, out_dir / "panel_base_vs_ar.jpg")           # (3)

    print("✓ Saved: points.jpg, ar_composite.jpg, panel_base_vs_ar.jpg")

# ================================== Main ==================================

def main():
    print("=" * 70)
    print("Part 2: Creative Applications of Homography")
    print("=" * 70)
    print("[1] Document Scanner")
    print("[2] AR Poster/Logo Insertion")
    choice = (input("Choose mode (1/2, default 2): ").strip() or "2")
    if choice == "1":
        document_scanner()
    else:
        ar_insert()

if __name__ == "__main__":
    main()
