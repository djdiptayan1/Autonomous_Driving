import cv2
import numpy as np
from sympy import fps
import utlis2
import time

curveList = []
avgVal = 10


def getLaneCurve(img, display=2):

    imgCopy = img.copy()
    imgResult = img.copy()
    #### STEP 1
    imgThres = utlis2.thresholding(img)

    #### STEP 2
    hT, wT, c = img.shape
    points = utlis2.valTrackbars()
    imgWarp = utlis2.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utlis2.drawPoints(imgCopy, points)

    #### STEP 3
    middlePoint, imgHist = utlis2.getHistogram(
        imgWarp, display=True, minPer=0.5, region=4
    )
    curveAveragePoint, imgHist = utlis2.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint

    #### SETP 4
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)

    curve = int(sum(curveList) / len(curveList))

    #### STEP 5
    if display != 0:
        imgInvWarp = utlis2.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0 : hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(
            imgResult,
            str(curve),
            (wT // 2 - 80, 85),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (255, 0, 255),
            3,
        )
        cv2.line(
            imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5
        )
        cv2.line(
            imgResult,
            ((wT // 2 + (curve * 3)), midY - 25),
            (wT // 2 + (curve * 3), midY + 25),
            (0, 255, 0),
            5,
        )
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(
                imgResult,
                (w * x + int(curve // 50), midY - 10),
                (w * x + int(curve // 50), midY + 10),
                (0, 0, 255),
                2,
            )
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = utlis2.stackImages(
            0.7, ([img, imgWarpPoints, imgWarp], [imgHist, imgLaneColor, imgResult])
        )
        cv2.imshow("ImageStack", imgStacked)
    elif display == 1:
        cv2.imshow("Resutlt", imgResult)

    #### NORMALIZATION
    curve = curve / 100
    if curve > 1:
        curve = 1
    if curve < -1:
        curve = -1

    # Check if curve is 0 (no lane detected), return 5 in that case
    if curve == 0:
        return 5

    return curve


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    # cap = cv2.resize(cap, (854, 480))
    initialTrackVals = [214, 20, 121, 221]
    utlis2.initializeTrackbars(initialTrackVals)

    frameCounter = 0
    while cap.isOpened():
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, frame = cap.read()
        frame = cv2.resize(frame, (854, 480))

        # Split the stereo frame into left and right images
        height, width, _ = frame.shape
        # print(height, "\t", width)
        width //= 2
        left_frame = frame[:, :width, :]
        right_frame = frame[:, width:, :]

        timer = cv2.getTickCount()

        curve = getLaneCurve(frame, display=2)
        print(f"Curve: {curve}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
