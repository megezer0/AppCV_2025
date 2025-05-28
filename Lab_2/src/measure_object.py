#!/usr/bin/env python3
"""
measure_object.py  – Estimate real-world distance between two user-clicked
points lying on the calibration board plane (checkerboard **or** ChArUco).

Usage:
    python3 src/measure_object.py --img data/my_photo.jpg \
                                  --board charuco        \
                                  --yaml captured_points/intrinsics.yml
    # afterwards: click P1, click P2, then press <ENTER>

Requires:
    • OpenCV ≥ 4.5  (with ArUco contrib)
    • Matplotlib   (for the simple GUI)
"""

import argparse, sys, cv2, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# ---------- CLI ----------
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('--img',   required=True,  help='Path to photograph')
ap.add_argument('--yaml',  default='captured_points/intrinsics.yml',
                help='YAML containing K and D')
ap.add_argument('--board', choices=['checker', 'charuco'], default='checker',
                help='Pattern used during calibration')
ap.add_argument('--cols',  type=int, default=9, help='# inner corners (checker) OR squares (ChArUco) along X')
ap.add_argument('--rows',  type=int, default=6, help='# along Y')
ap.add_argument('--sq',    type=float, default=0.03, help='Square size in metres')
args = ap.parse_args()

# ---------- load intrinsics ----------
with open(args.yaml, 'r') as f:
    intr = yaml.safe_load(f)
K   = np.asarray(intr['K']['data']).reshape(3,3)
D   = np.asarray(intr['D']['data']).reshape(-1,1)

# ---------- load image ----------
img = cv2.imread(args.img)
if img is None:
    sys.exit(f"Cannot read {args.img}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- detect board & estimate extrinsics ----------
if args.board == 'checker':
    pattern = (args.cols, args.rows)
    found, corners = cv2.findChessboardCorners(gray, pattern,
                     flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    if not found:
        sys.exit("Checkerboard not found.")
    cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    objp = np.zeros((args.rows*args.cols,3), np.float32)
    objp[:,:2] = np.indices((args.cols, args.rows)).T.reshape(-1,2)
    objp *= args.sq
elif args.board == 'charuco':
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    board      = cv2.aruco.CharucoBoard_create(args.cols, args.rows, args.sq, 0.75*args.sq, aruco_dict)
    cornersA, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if ids is None or len(ids) < 4:
        sys.exit("ChArUco markers not detected.")
    _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(cornersA, ids, gray, board)
    if ch_corners is None or len(ch_corners) < 4:
        sys.exit("Not enough ChArUco corners.")
    corners = ch_corners
    objp    = board.chessboardCorners[ch_ids.flatten()].astype(np.float32)
else:
    raise ValueError("Unsupported board type.")

_, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)

# Build homography:  H maps pixel → world-XY (Z=0) coordinates
R, _  = cv2.Rodrigues(rvec)
Rt    = np.hstack((R[:, :2], tvec))          # keep only r1 r2 t
H     = K @ Rt                               # 3×3
H_inv = np.linalg.inv(H)

def pix_to_world(pt):
    """Convert undistorted pixel (u,v) to 2-D world coords on the plane (metres)."""
    uv1 = np.array([pt[0], pt[1], 1.0])
    xn  = H_inv @ uv1
    xn /= xn[2]
    return xn[:2]   # (X, Y) on plane

# ---------- interactive click ----------
plt.figure(figsize=(6,4))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Click TWO points on the object; press Enter')
pts = plt.ginput(2, timeout=0)
plt.close()
if len(pts) != 2:
    sys.exit("Need exactly two clicks.")

# Undistort the clicked pixels before mapping
pts_ud = cv2.undistortPoints(np.array(pts, dtype=np.float32).reshape(-1,1,2), K, D, P=K)
pts_ud = pts_ud.reshape(-1,2)

p1_w, p2_w = pix_to_world(pts_ud[0]), pix_to_world(pts_ud[1])
dist = np.linalg.norm(p1_w - p2_w)

print(f"Distance = {dist*1000:.1f} mm   (Board: {args.board})")