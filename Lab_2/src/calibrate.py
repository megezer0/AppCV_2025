import numpy as np
import cv2
import yaml

data = np.load('captured_points/corners.npz', allow_pickle=True)

if 'objpoints' in data:
    # Checkerboard calibration
    objpoints = data['objpoints']
    imgpoints = data['imgpoints']
    img_shape = tuple(data['img_shape'])
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

elif 'corners' in data:
    # Charuco calibration
    corners = data['corners']
    ids = data['ids']
    img_shape = tuple(data['img_shape'])
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(5,7,0.03,0.022,aruco_dict)
    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(corners, ids, board, img_shape, None, None)

calibration_data = {'K': K.tolist(), 'dist': dist.tolist()}
with open('captured_points/intrinsics.yml', 'w') as f:
    yaml.dump(calibration_data, f)

print("Calibration complete. Intrinsics saved to captured_points/intrinsics.yml")
print(f"Mean reprojection error: {ret}")