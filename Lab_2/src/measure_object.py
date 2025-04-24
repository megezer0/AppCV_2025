import cv2, numpy as np, yaml

with open('captured_points/intrinsics.yml') as f:
    intrinsics = yaml.safe_load(f)

K = np.array(intrinsics['K'])
dist = np.array(intrinsics['dist'])

img = cv2.imread('data/metric_check.jpg')
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x,y])
        print(f"Point selected: {x},{y}")

cv2.imshow('Select two points', img)
cv2.setMouseCallback('Select two points', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

pts = np.array(points, dtype='float32').reshape(-1,1,2)
undistorted_pts = cv2.undistortPoints(pts, K, dist, P=K)
scale = 30  # mm per square (adjust as per board used)
distance = np.linalg.norm(undistorted_pts[0] - undistorted_pts[1]) * scale

print(f"Estimated length: {distance:.2f} mm")