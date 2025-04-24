import cv2
import cv2.aruco as aruco
import numpy as np
import glob

BOARD_TYPE = "charuco"
images = sorted(glob.glob('data/img_*.jpg'))

if BOARD_TYPE == "checker":
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    objpoints, imgpoints = [], []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
            imgpoints.append(corners2)

    np.savez('captured_points/corners.npz', objpoints=objpoints, imgpoints=imgpoints, img_shape=gray.shape[::-1])

elif BOARD_TYPE == "charuco":
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard_create(5, 7, 0.03, 0.022, aruco_dict)
    all_corners, all_ids = [], []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        if ids is not None:
            _, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ch_corners is not None:
                all_corners.append(ch_corners)
                all_ids.append(ch_ids)

    np.savez('captured_points/corners.npz', corners=all_corners, ids=all_ids, img_shape=gray.shape[::-1])