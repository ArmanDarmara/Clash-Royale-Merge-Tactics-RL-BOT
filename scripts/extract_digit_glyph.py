"""
Isolate a single clean digit glyph from a badge region via HSV white-fill
masking + largest-contour selection, and save it as a template for
match_icon()-style digit matching. One-off tool used to seed
data/reference_digits/ from the calibration captures.

Usage: python3 extract_digit_glyph.py <image> <x0> <y0> <x1> <y1> <out_name>
  region is (x0,y0,x1,y1) as fractions of the image.
"""
import sys
import cv2
import numpy as np

def extract(image_path, region, out_path, upscale=6):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region
    sub = img[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
    sub = cv2.resize(sub, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LANCZOS4)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 190), (180, 60, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("no contours found")
        return None
    # pick the largest contour whose box isn't touching the crop edge (that's
    # usually decorative chrome bleeding in, not the digit itself)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: -b[2]*b[3])
    bx, by, bw, bh = boxes[0]
    pad = 12
    y0p, y1p = max(0, by-pad), by+bh+pad
    x0p, x1p = max(0, bx-pad), bx+bw+pad
    glyph = sub[y0p:y1p, x0p:x1p]
    cv2.imwrite(out_path, glyph)
    print(f"saved {out_path} from box {(bx,by,bw,bh)}")
    return out_path

if __name__ == "__main__":
    image_path = sys.argv[1]
    x0, y0, x1, y1 = map(float, sys.argv[2:6])
    out_path = sys.argv[6]
    extract(image_path, (x0, y0, x1, y1), out_path)
