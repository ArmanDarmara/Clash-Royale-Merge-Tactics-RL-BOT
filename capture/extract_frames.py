"""
Pull frames out of a screen recording for analysis. This one runs fine in
Cowork's device shell (or anywhere) since it's pure file/video processing --
no GUI/display access needed, unlike window_capture.py / input_control.py.

Uses seek-based sampling (jump to each target frame) rather than decoding
every frame sequentially -- much faster for sparse sampling on a long video.

Usage: python3 extract_frames.py <video_path> <out_dir> [fps] [start_sec] [end_sec]
"""
import sys
import os
import cv2


def extract(video_path, out_dir, sample_fps=1, start_sec=None, end_sec=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps

    start_sec = start_sec if start_sec is not None else 0
    end_sec = end_sec if end_sec is not None else duration

    saved = 0
    t = start_sec
    step = 1.0 / sample_fps
    while t < end_sec:
        frame_idx = int(t * src_fps)
        if frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(os.path.join(out_dir, f"t{t:07.2f}.png"), frame)
            saved += 1
        t += step
    cap.release()
    print(f"Saved {saved} frames to {out_dir} (video duration {duration:.1f}s)")


if __name__ == "__main__":
    video_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "../data/recordings/frames"
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 1
    start_sec = float(sys.argv[4]) if len(sys.argv) > 4 else None
    end_sec = float(sys.argv[5]) if len(sys.argv) > 5 else None
    extract(video_path, out_dir, fps, start_sec, end_sec)
