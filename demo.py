#!/usr/bin/env python3
"""InstantHMR demo: people in an image, a video or a live camera, shown in the Rerun viewer.

    python demo.py --image photo.jpg
    python demo.py --video clip.mp4 --pose-filter one-euro
    python demo.py --camera 0 --model small --fit 1 --device tensorrt
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from instanthmr import DEFAULT_FIT, DEFAULT_MODEL, MODELS, PosePipeline
from instanthmr.smoothing import TemporalSmoother
from instanthmr.visualizer import RerunVisualizer

# `--quality` shortcut -> released model name. `best` and `medium` are the same graph: the highest-accuracy
# tier (EMDB-1 PA-MPJPE 44.4 mm with --fit 2), non-commercial (SAM License). `commercial` is the
# commercially usable tier (68.3 mm).
QUALITY_ALIASES = {
    "best": "medium",
    "medium": "medium",
    "commercial": "medium-commercial",
    "small": "small",
    "tiny": "tiny",
    "nano": "nano",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="image file")
    src.add_argument("--video", help="video file")
    src.add_argument("--camera", type=int, metavar="INDEX", help="live camera, e.g. 0")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"{', '.join(MODELS)}, or a path to an .onnx file (default: {DEFAULT_MODEL})")
    p.add_argument("--quality", choices=tuple(QUALITY_ALIASES),
                   help="shortcut for --model: best/medium (non-commercial, highest accuracy), "
                        "commercial, small, tiny, nano (overrides --model; the -fitN graph is picked by --fit)")
    p.add_argument("--fit", type=int, default=DEFAULT_FIT, choices=(0, 1, 2),
                   help="kinematic fit on 1 or 2 crops per person, 0 for the network alone (default: "
                        f"{DEFAULT_FIT}; ignored for an .onnx path)")
    p.add_argument("--device", default="cuda", choices=("cuda", "tensorrt", "cpu", "coreml"))
    p.add_argument("--max-persons", type=int, default=4)
    p.add_argument("--detector", default="medium", choices=("nano", "small", "medium", "base", "large"),
                   help="RF-DETR size (default: medium)")
    p.add_argument("--det-confidence", type=float, default=0.5)
    p.add_argument("--detector-stride", type=int, default=1, metavar="N",
                   help="run the detector every N frames and reuse its boxes in between")
    p.add_argument("--focal", type=float, help="camera focal length in pixels, if known")
    p.add_argument("--pose-filter", choices=("one-euro", "alpha-beta"),
                   help="video: filter each person's pose across frames (more accurate and steadier)")
    p.add_argument("--smooth", action="store_true",
                   help="video: smooth the keypoints and mesh instead (steadiest, but lags on fast motion)")
    p.add_argument("--no-mesh", action="store_true", help="show the skeleton only")
    p.add_argument("--mesh-alpha", type=float, default=0.35, help="mesh opacity (default: 0.35)")
    p.add_argument("--no-floor", action="store_true", help="3D: hide the ground grid drawn under the people")
    p.add_argument("--viewer-grid", action="store_true",
                   help="3D: show the Rerun viewer's built-in line grid (hidden by default)")
    p.add_argument("--show-camera-3d", action="store_true",
                   help="3D: also draw the camera pinhole and the frame's image plane inside the 3D view "
                        "(hidden by default, the 2D 'Camera' panel is unaffected)")
    p.add_argument("--floor-size", type=float, default=4.0, metavar="METRES",
                   help="3D: side length of the ground grid (default: 4.0)")
    p.add_argument("--save-rrd", metavar="PATH", help="also save the Rerun recording")
    p.add_argument("--no-viewer", action="store_true", help="do not open the Rerun viewer")
    p.add_argument("--audio", action="store_true",
                   help="video: start with the clip's sound already on (the control panel can flip it either "
                        "way; Rerun 0.31 has no audio archetype, so the sound cannot be embedded in the .rrd)")
    p.add_argument("--no-audio", dest="audio", action="store_false",
                   help="video: start muted, the panel's sound switch stays available (default)")
    p.add_argument("--no-panel", action="store_true",
                   help="do not open the small control panel (audio and 3D switches then only come from flags)")
    return p.parse_args()


def frames(args):
    """(index, timestamp in s, RGB frame) from the chosen source."""
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            sys.exit(f"cannot read image {args.image}")
        yield 0, 0.0, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video or f'camera {args.camera}'}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, i = time.perf_counter(), 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            t = i / fps if args.video else time.perf_counter() - start
            yield i, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    if args.quality:
        args.model = QUALITY_ALIASES[args.quality]
    fit = f", kinematic fit on {args.fit} crop{'s' if args.fit > 1 else ''}" if args.fit else ""
    print(f"loading InstantHMR ({args.model}{fit}) on {args.device} and RF-DETR {args.detector}...")
    pipeline = PosePipeline(args.model, fit=args.fit, device=args.device, detector=args.detector,
                            det_confidence=args.det_confidence, max_persons=args.max_persons,
                            detector_stride=args.detector_stride, focal=args.focal, pose_filter=args.pose_filter)
    print(f"InstantHMR runs on {pipeline.hmr.provider}")
    viewer = RerunVisualizer(None if args.no_mesh else pipeline.hmr.faces, args.mesh_alpha,
                             spawn=not args.no_viewer, save_path=args.save_rrd,
                             floor=not args.no_floor, viewer_grid=args.viewer_grid,
                             floor_size=args.floor_size, camera_in_3d=args.show_camera_3d)
    smoother = TemporalSmoother() if args.smooth else None
    player = panel = None
    now = [0.0]
    if args.video:
        from instanthmr.audio import has_audio, open_clip_audio

        has_track = has_audio(args.video)
        if has_track:
            player = open_clip_audio(args.video)
            if player is not None:
                player.set_enabled(bool(args.audio))
    else:
        has_track = False
    if not args.no_panel:
        from instanthmr.control_panel import ControlPanel

        panel = ControlPanel(player, viewer if not args.no_viewer else None,
                             clock=lambda: now[0], has_audio=has_track)

    history = []
    try:
        for i, t, rgb in frames(args):
            now[0] = t
            if panel is not None:
                panel.poll()
            sync = (player is not None and player.enabled and panel is not None
                    and not panel.closed and bool(panel.sync.get()))
            if sync:
                # Audio is the master clock: hold the frames back until the sound reaches them, and jump
                # the sound forward if a stall let it run ahead. With sync off the loop runs at full speed.
                if player.cursor > t + 0.25:
                    player.seek(t)
                while player.cursor + 0.02 < t and not player.at_end and not panel.closed:
                    panel.poll()
                    time.sleep(0.004)
            result = pipeline(rgb, t)
            persons = smoother(result.persons, t) if smoother else result.persons
            timings = {"detector": result.detector_ms, "InstantHMR": result.hmr_ms}
            viewer.log(i, rgb, persons, timings)
            history.append((result.detector_ms, result.hmr_ms))
            if i % 30 == 0:
                print(f"frame {i:5d}  {len(persons)} person(s)  detector {result.detector_ms:5.1f} ms  "
                      f"InstantHMR {result.hmr_ms:5.1f} ms")
    except KeyboardInterrupt:
        pass
    if panel is not None:
        panel.close()
    elif player is not None:
        player.stop()
    if len(history) > 1:
        det, hmr = np.median(np.array(history[1:]), axis=0)
        print(f"median over {len(history) - 1} frames: detector {det:.1f} ms, InstantHMR {hmr:.1f} ms")


if __name__ == "__main__":
    main()
