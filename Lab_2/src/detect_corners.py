#!/usr/bin/env python3
import os, glob, cv2, numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# BOARD_TYPE: "checker" or "charuco"
BOARD_TYPE = "checker"

# For a checkerboard with NUM_SQUARES_X × NUM_SQUARES_Y squares,
# there are (NUM_SQUARES_X - 1) × (NUM_SQUARES_Y - 1) interior corners.
NUM_SQUARES_X = 10   # e.g. you printed 10 squares across
NUM_SQUARES_Y = 7    # e.g. you printed  7 squares down

# Chessboard square side length (in meters; used later for measurement).
SQUARE_SIZE_M = 0.03

# Derived parameters for checkerboard
PATTERN_SIZE = (NUM_SQUARES_X - 1, NUM_SQUARES_Y - 1)  # (cols, rows)
NUM_CORNERS = PATTERN_SIZE[0] * PATTERN_SIZE[1]
# -----------------------------------------------------------------------------

# Build an explicit path to the top‐level data/ folder
BASE_DIR = os.path.dirname(__file__) + "/.."
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR  = os.path.join(BASE_DIR, 'captured_points')
os.makedirs(OUT_DIR, exist_ok=True)

images = sorted(glob.glob(os.path.join(DATA_DIR, 'img_*.jpg')))

if BOARD_TYPE == "checker":
    # Prepare object‐space points for a planar grid at Z=0
    # Shape: (NUM_CORNERS, 3)
    objp = np.zeros((NUM_CORNERS, 3), np.float32)
    # Fill in (x,y) coordinates: 0,1,2,…,PATTERN_SIZE[0]-1  and same for Y
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0],
                           0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    # Scale by actual square size:
    objp *= SQUARE_SIZE_M  

    objpoints, imgpoints = [], []
    img_shape = None

    valid_files = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            flags=(cv2.CALIB_CB_ADAPTIVE_THRESH
                   + cv2.CALIB_CB_NORMALIZE_IMAGE)
        )
        if not ret:
            continue

        # Refine corner positions to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners2)
        valid_files.append(fname)

    # Save to disk
    np.savez(
        os.path.join(OUT_DIR, 'corners.npz'),
        valid_files=valid_files,
        objpoints=objpoints,
        imgpoints=imgpoints,
        img_shape=img_shape
    )

elif BOARD_TYPE == "charuco":
    # (unchanged CharUco code...)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(5, 7, 0.03, 0.022, aruco_dict)

    all_corners, all_ids = [], []
    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if ids is None:
            continue

        _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if ch_corners is not None:
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)

    np.savez(
        os.path.join(OUT_DIR, 'corners.npz'),
        corners=all_corners,
        ids=all_ids,
        img_shape=img_shape
    )

print("✅ Corner detection complete.")