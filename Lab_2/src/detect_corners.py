#!/usr/bin/env python3
"""
detect_corners.py -- Extract checkerboard or ChArUco corners from calibration images

Usage:
    python3 src/detect_corners.py --board checker
    python3 src/detect_corners.py --board charuco

Output:
    Saves corner detections to `captured_points/corners.npz` for later use in calibration.
"""

import os, glob, cv2, numpy as np
import argparse

# ------------------- CLI ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('--board', choices=['checker', 'charuco'], required=True)
args = parser.parse_args()
BOARD_TYPE = args.board

# ---------------- CONFIG ---------------------
NUM_SQUARES_X = 10
NUM_SQUARES_Y = 7
SQUARE_SIZE_M = 0.03

PATTERN_SIZE = (NUM_SQUARES_X - 1, NUM_SQUARES_Y - 1)
NUM_CORNERS = PATTERN_SIZE[0] * PATTERN_SIZE[1]

BASE_DIR = os.path.dirname(__file__) + "/.."
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR  = os.path.join(BASE_DIR, 'captured_points')
os.makedirs(OUT_DIR, exist_ok=True)

images = sorted(glob.glob(os.path.join(DATA_DIR, 'img_*.jpg')))
assert len(images) > 0, "No calibration images found."

# ---------------- DETECTION -------------------
if BOARD_TYPE == "checker":
    objp = np.zeros((NUM_CORNERS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    objpoints, imgpoints = [], []
    valid_files = []
    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ret:
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)
        valid_files.append(fname)

    np.savez(
        os.path.join(OUT_DIR, 'corners.npz'),
        valid_files=valid_files,
        objpoints=objpoints,
        imgpoints=imgpoints,
        img_shape=img_shape
    )

elif BOARD_TYPE == "charuco":
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(5, 7, 0.03, 0.022, aruco_dict)

    all_corners, all_ids = [], []
    img_shape = None
    valid_files = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if ids is None:
            continue

        _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if ch_corners is None or len(ch_corners) < 4:
            continue

        all_corners.append(ch_corners)
        all_ids.append(ch_ids)
        valid_files.append(fname)

    np.savez(
        os.path.join(OUT_DIR, 'corners.npz'),
        valid_files=valid_files,
        corners=all_corners,
        ids=all_ids,
        img_shape=img_shape
    )

print(f"✅ Corner detection complete. {len(valid_files)} valid frames saved.")
print(f"→ Output: {os.path.join(OUT_DIR, 'corners.npz')}")