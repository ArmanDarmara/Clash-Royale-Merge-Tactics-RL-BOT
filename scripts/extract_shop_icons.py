"""
Scan extracted frames, crop the 3 shop-card portrait slots from each,
dedupe via a simple average-hash, and save one representative full-res
crop per unique icon to data/reference_cards/unlabeled/.

Pure PIL, no GUI/display needed -- safe to run in the Linux VM.
"""
import glob
import os
from PIL import Image

FRAMES_DIR = "../data/recordings/frames"
OUT_DIR = "../data/reference_cards/unlabeled"

X0, X1 = 0.275, 0.685
Y0, Y1 = 0.893, 0.995

def ahash(img, size=8):
    im = img.convert("L").resize((size, size), Image.LANCZOS)
    pixels = list(im.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return bits

def hamming(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "t*.png")))
    print(f"Scanning {len(frames)} frames...")

    known_hashes = []
    saved_count = 0
    checked = 0

    for fp in frames:
        try:
            im = Image.open(fp)
        except Exception:
            continue
        w, h = im.size
        x0, x1 = int(w * X0), int(w * X1)
        y0, y1 = int(h * Y0), int(h * Y1)
        card_w = (x1 - x0) / 3
        for i in range(3):
            cx0 = int(x0 + i * card_w)
            cx1 = int(x0 + (i + 1) * card_w)
            crop = im.crop((cx0, y0, cx1, y1))
            gray = crop.convert("L")
            hist = gray.histogram()
            total = sum(hist)
            modal_frac = max(hist) / total if total else 1
            if modal_frac > 0.6:
                continue
            h_ = ahash(crop)
            checked += 1
            match = None
            for kh, kpath in known_hashes:
                if hamming(h_, kh) <= 6:
                    match = kpath
                    break
            if match is None:
                out_path = os.path.join(OUT_DIR, f"card_{saved_count:03d}.png")
                crop.save(out_path)
                known_hashes.append((h_, out_path))
                saved_count += 1

    print(f"Checked {checked} card crops, saved {saved_count} unique icons to {OUT_DIR}")

if __name__ == "__main__":
    main()
