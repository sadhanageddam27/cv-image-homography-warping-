# homography_part1.py — stable exact-4 flow with reorder prompt (no hidden repeats)

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ---------------- I/O ----------------
def load_rgb(path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def show_grid(img, title):
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, img.shape[1], 50))
    ax.set_yticks(np.arange(0, img.shape[0], 50))
    plt.grid(True, alpha=0.35)
    plt.title(title)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.show(block=True)
    plt.close('all')

# ------------- click EXACTLY 4 points (undo, enter) -------------
def click_points_exact4(img):
    print("\nClick EXACTLY 4 points (any order for now).")
    print("Press ENTER when done, 'u' to undo last point.")
    pts = []
    fig, ax = plt.subplots(figsize=(10,7))
    ax.imshow(img); ax.grid(True, alpha=0.25)
    ax.set_title("Pick 4 points")
    scat = ax.scatter([], [], c='r', s=60)
    lines, = ax.plot([], [], 'y-', linewidth=2)
    txt = ax.text(10,10,"Points: 0/4", color='yellow',
                  bbox=dict(facecolor='black', alpha=0.5), fontsize=12)

    def redraw():
        if pts:
            arr = np.array(pts)
            scat.set_offsets(arr)
            if arr.shape[0] >= 2:
                cyc = np.vstack([arr, arr[0]])
                lines.set_data(cyc[:,0], cyc[:,1])
        txt.set_text(f"Points: {len(pts)}/4")
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax: return
        if len(pts) < 4:
            pts.append([ev.xdata, ev.ydata])
            redraw()

    def on_key(ev):
        if ev.key in ('u','U','backspace'):
            if pts:
                pts.pop()
                if len(pts) < 2:
                    lines.set_data([], [])
                redraw()
        elif ev.key == 'enter':
            if len(pts) == 4:
                plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    plt.close(fig)
    if len(pts) != 4:
        raise RuntimeError("You must select exactly 4 points.")
    return np.array(pts, dtype=np.float64)

# ------------- utilities -------------
def too_close(pts, min_dist=5.0):
    """Return True if any pair of points is closer than min_dist pixels."""
    for i in range(4):
        for j in range(i+1,4):
            if np.linalg.norm(pts[i]-pts[j]) < min_dist:
                return True
    return False

def polygon_area(quad):
    x, y = quad[:,0], quad[:,1]
    return 0.5*abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

# ------------- big, safe, high-contrast labels -------------
def draw_points_big(img: np.ndarray, pts: np.ndarray, title: str) -> np.ndarray:
    vis = img.copy()
    H, W = vis.shape[:2]
    diag = (H**2 + W**2)**0.5
    r_out = max(6, int(diag * 0.007))
    r_in  = max(4, int(diag * 0.0045))
    cross = max(10, int(diag * 0.012))
    thick = max(2, int(diag * 0.0025))
    fscale= max(0.6, diag * 0.0012)
    pad   = max(6, int(diag * 0.005))
    palette=[(255,255,0),(0,255,255),(255,0,255),(0,255,0),(255,128,0),(255,0,0),(0,0,255)]

    def safe_box(xp, yp, tw, th):
        tl_x = xp + pad; tl_y = yp - th - pad
        br_x = tl_x + tw + pad; br_y = tl_y + th + pad
        if br_x > W-1: tl_x = max(0, xp - pad - tw - pad); br_x = tl_x + tw + pad
        if tl_x < 0:   tl_x = 0; br_x = tl_x + tw + pad
        if tl_y < 0:   tl_y = yp + pad; br_y = tl_y + th + pad
        if br_y > H-1: tl_y = max(0, H-1 - (th + pad)); br_y = tl_y + th + pad
        return (int(tl_x),int(tl_y)), (int(br_x),int(br_y))

    for i,(x,y) in enumerate(pts, start=1):
        xp, yp = int(round(x)), int(round(y))
        color = palette[(i-1)%len(palette)]
        cv2.circle(vis,(xp,yp),r_out,(255,255,255),thick,cv2.LINE_AA)
        cv2.circle(vis,(xp,yp),r_in,color,-1,cv2.LINE_AA)
        cv2.line(vis,(xp-cross,yp),(xp+cross,yp),(0,0,0),thick,cv2.LINE_AA)
        cv2.line(vis,(xp,yp-cross),(xp,yp+cross),(0,0,0),thick,cv2.LINE_AA)
        label=f"P{i}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,fscale,thick)
        tl, br = safe_box(xp, yp, tw, th)
        cv2.rectangle(vis, tl, br, (255,255,255), max(2,thick))
        cv2.rectangle(vis, (tl[0]+2,tl[1]+2),(br[0]-2,br[1]-2), (0,0,0), -1)
        cv2.putText(vis, label, (tl[0]+pad//2, br[1]-pad//2),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, (0,255,255), max(2,thick), cv2.LINE_AA)

    plt.figure(figsize=(10,7)); plt.imshow(vis); plt.axis("off"); plt.title(title)
    plt.show(block=True); plt.close('all')
    return vis

# ------------- normalized DLT + warp -------------
def _normalize(pts):
    c = pts.mean(axis=0)
    shifted = pts - c
    mean_d = np.mean(np.linalg.norm(shifted, axis=1))
    s = np.sqrt(2) / (mean_d + 1e-12)
    T = np.array([[s,0,-s*c[0]],[0,s,-s*c[1]],[0,0,1]], dtype=np.float64)
    pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
    pts_n = (T @ pts_h.T).T[:, :2]
    return T, pts_n

def estimate_H_normalized(src, dst):
    T1,p1=_normalize(src); T2,p2=_normalize(dst)
    A=[]
    for (x,y),(u,v) in zip(p1,p2):
        A.append([-x,-y,-1, 0, 0, 0, u*x, u*y, u])
        A.append([ 0, 0, 0,-x,-y,-1, v*x, v*y, v])
    A=np.array(A, dtype=np.float64)
    _,_,Vt=np.linalg.svd(A)
    Hn=Vt[-1].reshape(3,3)
    H=np.linalg.inv(T2) @ Hn @ T1
    return H/H[2,2]

# ------------- per-image pipeline -------------
def process_image(img_path: Path, out_root: Path):
    img = load_rgb(img_path)
    show_grid(img, f"Original grid – {img_path.name}")

    while True:
        pts4 = click_points_exact4(img)

        # 1) fail fast if any duplicate/too-close clicks
        if too_close(pts4, min_dist=5.0):
            print("\n[!] Two points were too close (duplicates). Please re-pick 4 points.")
            continue

        # 2) preview the clicked order and labels
        preview_clicked = draw_points_big(img, pts4, "Preview: your 4 clicks in order (P1..P4)")

        # 3) ask if this order already corresponds to TL,TR,BR,BL
        ok = input("Use this order as TL,TR,BR,BL? [y/n]: ").strip().lower()
        if ok == 'y':
            src4 = pts4.copy()
        else:
            # let user specify the mapping to TL,TR,BR,BL
            print("Enter a permutation like '2 3 4 1' meaning: TL=P2, TR=P3, BR=P4, BL=P1")
            while True:
                try:
                    perm = input("Indices for TL TR BR BL: ").strip().split()
                    if len(perm) != 4: raise ValueError
                    idx = [int(k) for k in perm]
                    if sorted(idx) != [1,2,3,4]: raise ValueError
                    src4 = pts4[np.array([k-1 for k in idx])]
                    break
                except Exception:
                    print("Invalid input. Please enter four numbers 1..4 exactly once each.")

        # 4) sanity check area
        if polygon_area(src4) < 1.0:
            print("\n[!] The quadrilateral area is too small (nearly collinear). Re-pick.")
            continue

        # 5) final visualization of the ordered quad
        overlay = img.copy()
        p = src4.astype(int)
        for i in range(4):
            cv2.line(overlay, tuple(p[i]), tuple(p[(i+1)%4]),
                     (0,255,0), 3, cv2.LINE_AA)
        plt.figure(figsize=(8,6)); plt.imshow(overlay); plt.axis('off')
        plt.title("Final quad (TL→TR→BR→BL)"); plt.show(block=True); plt.close('all')

        confirm = input("Proceed with these corners? [y/n]: ").strip().lower()
        if confirm == 'y':
            break
        else:
            print("Okay, let's re-pick the 4 points.\n")

    # Manual output size
    print("\nEnter desired output size (you control aspect):")
    out_w = int(input("  Width  (e.g., 800): "))
    out_h = int(input("  Height (e.g., 600): "))

    dst4 = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float64)

    # H + warp
    H = estimate_H_normalized(src4, dst4)
    rectified = cv2.warpPerspective(img, H.astype(np.float64),
                                    (out_w, out_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

    # Label destination corners
    rect_lab = draw_points_big(rectified, dst4, "Rectified + destination corners (P1–P4)")

    # Save
    out_dir = out_root / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rectified).save(out_dir/"rectified.jpg")
    Image.fromarray(rect_lab).save(out_dir/"rectified_with_corners.jpg")
    Image.fromarray(preview_clicked).save(out_dir/"original_points.jpg")
    np.save(out_dir/"H.npy", H)

    with open(out_dir/"src_points.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["label","x","y"])
        for i,(x,y) in enumerate(src4,1): w.writerow([f"P{i}",float(x),float(y)])
    with open(out_dir/"dst_points.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["label","x","y"])
        for i,(x,y) in enumerate(dst4,1): w.writerow([f"P{i}",float(x),float(y)])

    print(f"✓ Saved → {out_dir}\n")

# ------------- main -------------
def main():
    print("="*70)
    print("PART 1 — Homography Rectification (exact-4 with reorder/confirm)")
    print("="*70)

    root = Path(__file__).parent
    src_folder = root / "part1_images"
    imgs = sorted([p for p in src_folder.iterdir()
                   if p.suffix.lower() in (".jpg",".jpeg",".png")])
    if not imgs:
        raise RuntimeError("No images in part1_images/")

    out_root = root / "results_part1"; out_root.mkdir(exist_ok=True)

    for i,p in enumerate(imgs,1):
        print("-"*65)
        print(f"[{i}/{len(imgs)}] {p.name}")
        print("-"*65)
        process_image(p, out_root)

    print("="*70)
    print("✓ ALL DONE — see results_part1/")
    print("="*70)

if __name__ == "__main__":
    main()
