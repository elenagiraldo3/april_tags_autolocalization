import cv2
import numpy as np
import matplotlib.pyplot as ppl
import pupil_apriltags as apriltag


def PolyArea2D(pts):
    l = np.hstack([pts, np.roll(pts, -1, axis=0)])
    a = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in l))
    return a


def plotCamera3D(Cesc, rvec, ax=None):
    if ax is None:
        ax = ppl.axes(projection='3d')
    point = ax.scatter3D(Cesc[0], Cesc[1], Cesc[2], 'k', c='red')
    R, _ = cv2.Rodrigues(rvec)

    p1_cam = [-20, 20, 50]
    p2_cam = [20, 20, 50]
    p3_cam = [20, -20, 50]
    p4_cam = [-20, -20, 50]

    p1_esc = R.T @ p1_cam + Cesc
    p2_esc = R.T @ p2_cam + Cesc
    p3_esc = R.T @ p3_cam + Cesc
    p4_esc = R.T @ p4_cam + Cesc
    camera_plot = [ax.plot3D((Cesc[0], p1_esc[0]), (Cesc[1], p1_esc[1]), (Cesc[2], p1_esc[2]), '-k'),
                   ax.plot3D((Cesc[0], p2_esc[0]), (Cesc[1], p2_esc[1]), (Cesc[2], p2_esc[2]), '-k'),
                   ax.plot3D((Cesc[0], p3_esc[0]), (Cesc[1], p3_esc[1]), (Cesc[2], p3_esc[2]), '-k'),
                   ax.plot3D((Cesc[0], p4_esc[0]), (Cesc[1], p4_esc[1]), (Cesc[2], p4_esc[2]), '-k'),
                   ax.plot3D((p1_esc[0], p2_esc[0]), (p1_esc[1], p2_esc[1]), (p1_esc[2], p2_esc[2]), '-k'),
                   ax.plot3D((p2_esc[0], p3_esc[0]), (p2_esc[1], p3_esc[1]), (p2_esc[2], p3_esc[2]), '-k'),
                   ax.plot3D((p3_esc[0], p4_esc[0]), (p3_esc[1], p4_esc[1]), (p3_esc[2], p4_esc[2]), '-k'),
                   ax.plot3D((p4_esc[0], p1_esc[0]), (p4_esc[1], p1_esc[1]), (p4_esc[2], p1_esc[2]), '-k')]

    return camera_plot, point


def getCamera3D(rvec, tvec):
    # Centro óptico de la cámara como un punto 3D expresado en el sistema de la escena
    # t = -R @ Cesc => Cesc = -R^-1 @ t, pero R^-1 = R.T => Cesc = -R.T @ t
    R, _ = cv2.Rodrigues(rvec)
    Cesc = (-R.T @ tvec).reshape(3)

    return Cesc


npz_file = "calibration.npz"
tagsize = 160.0
family = "tag36h11"
camera = 1
ids = [0, 1, 2]
objectPoints = [np.array([[0., 0., 0.], [tagsize, 0., 0.], [tagsize, tagsize, 0.], [0., tagsize, 0.]]),
                np.array(
                    [[500., 0., 0.], [500., -tagsize, 0.], [500. + tagsize, -tagsize, 0.], [500. + tagsize, 0., 0.]]),
                np.array(
                    [[0., 560., 93.], [tagsize, 560., 93.], [tagsize, 560., 93. + tagsize], [0., 560., 93. + tagsize]])]

with np.load(npz_file) as data:
    intrinsics = data['intrinsics']
    dist_coeffs = data['dist_coeffs']

vs = cv2.VideoCapture(camera)
detector = apriltag.Detector(families=family)
fig = ppl.figure(figsize=(3, 3))
axes = ppl.axes(projection='3d')
axes.set_xlabel('X (mm)')
axes.set_ylabel('Y (mm)')
axes.set_zlabel('Z (mm)')

for objectPoint in objectPoints:
    axes.scatter3D(objectPoint[0, 0], objectPoint[0, 1], objectPoint[0, 2], '-k', c='blue')

    axes.plot3D((objectPoint[0, 0], objectPoint[1, 0]), (objectPoint[0, 1], objectPoint[1, 1]),
                (objectPoint[0, 2], objectPoint[1, 2]), '-g')
    axes.plot3D((objectPoint[1, 0], objectPoint[2, 0]), (objectPoint[1, 1], objectPoint[2, 1]),
                (objectPoint[1, 2], objectPoint[2, 2]), '-g')
    axes.plot3D((objectPoint[2, 0], objectPoint[3, 0]), (objectPoint[2, 1], objectPoint[3, 1]),
                (objectPoint[2, 2], objectPoint[3, 2]), '-g')
    axes.plot3D((objectPoint[3, 0], objectPoint[0, 0]), (objectPoint[3, 1], objectPoint[0, 1]),
                (objectPoint[3, 2], objectPoint[0, 2]), '-g')

camera_points = []

while vs.isOpened():
    lines = []
    ret, image = vs.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)
    coord_fusion = []
    angle_fusion = []
    areas = []
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        imagePoints = r.corners

        ptA, ptB, ptC, ptD = imagePoints
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        areas.append((PolyArea2D(imagePoints)))

        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)

        # draw the left-down (x, y)-coordinates of the AprilTag
        cv2.circle(image, ptA, 5, (255, 0, 0), -1)

        # draw the tag id on the image
        tagid = "tag_id = " + str(r.tag_id)
        cv2.putText(image, tagid, (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, rotation, translation = cv2.solvePnP(objectPoints[r.tag_id], imagePoints, intrinsics, dist_coeffs)

        camera = getCamera3D(rotation, translation)
        coord_fusion.append(camera)
        angle_fusion.append(rotation)

    if len(results) > 0:
        total_weight = np.sum(areas)
        ratio = []
        for area in areas:
            ratio.append(area / total_weight)
        ratio = np.array(ratio)
        coord_fusion = np.array(coord_fusion)
        angle_fusion = np.array(angle_fusion)
        camera = np.array([0, 0, 0])
        sin_angle = np.array([0, 0, 0])
        cos_angle = np.array([0, 0, 0])
        for i in range(len(ratio)):
            camera = camera + (coord_fusion[i] * ratio[i])
            sin_angle = sin_angle + (np.sin(angle_fusion[i]).reshape(3) * ratio[i])
            cos_angle = cos_angle + (np.cos(angle_fusion[i]).reshape(3) * ratio[i])
        angle = np.arctan(sin_angle / cos_angle)
        angle[0] = angle[0] - np.pi
        lines, camera_point = plotCamera3D(camera, angle, axes)
        camera_points.append(camera_point)

    ppl.pause(0.0000000001)

    for line in lines:
        axes.lines.remove(line[0])
    del lines

    if len(camera_points) > 15:
        camera_points[0].remove()
        camera_points = camera_points[1:]

    cv2.imshow("camera", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ppl.show()
vs.release()
cv2.destroyAllWindows()
