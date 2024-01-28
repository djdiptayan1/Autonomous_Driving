import cv2
import numpy as np
from sympy import fps
import utils
import time

curveList = []
avgVal = 10


def getLaneCurve(image, display=2):
    imgResult = image.copy()
    imgThresh = utils.thresholding(image)

    h, w, c = image.shape
    points = utils.valTrackbars()
    imgWarp = utils.warpImg(imgThresh, points, w, h)
    imgWarpPoints = utils.drawPoints(image.copy(), points)

    # GET HISTOGRAM
    minPoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAvgPoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.9)

    curveRaw = curveAvgPoint - minPoint

    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)

    curve = int(sum(curveList) / len(curveList))

    if display != 0:
        imgInvWarp = utils.warpImg(imgWarp, points, w, h, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0 : h // 3, 0:w] = 0, 0, 0
        imgLaneColor = np.zeros_like(image)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(
            imgResult,
            str(curve),
            (w // 2 - 80, 85),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (255, 0, 255),
            3,
        )
        cv2.line(
            imgResult, (w // 2, midY), (w // 2 + (curve * 3), midY), (255, 0, 255), 5
        )
        cv2.line(
            imgResult,
            ((w // 2 + (curve * 3)), midY - 25),
            (w // 2 + (curve * 3), midY + 25),
            (0, 255, 0),
            5,
        )
        for x in range(-30, 30):
            w = w // 20
            cv2.line(
                imgResult,
                (w * x + int(curve // 50), midY - 10),
                (w * x + int(curve // 50), midY + 10),
                (0, 0, 255),
                2,
            )
        # FPS
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # cv2.putText(
        #     imgResult,
        #     "FPS " + str(int(fps)),
        #     (20, 40),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (230, 50, 50),
        #     3,
        # )
    if display == 2:
        imgStacked = utils.stackImages(
            0.7, ([image, imgWarpPoints, imgWarp], [imgHist, imgLaneColor, imgResult])
        )
        cv2.imshow("ImageStack", imgStacked)
    elif display == 1:
        cv2.imshow("Resutlt", imgResult)

    # # Check if imgThresh is grayscale
    # if len(imgThresh.shape) == 2:
    #     # Convert imgThresh to color
    #     imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR)

    # # Resize imgHist to match imgThresh
    # imgHist = cv2.resize(imgHist, (imgThresh.shape[1], imgThresh.shape[0]))

    # # Now you can merge the images
    # merged_abhi = cv2.addWeighted(imgThresh, 1, imgHist, 0.5, 0)

    # cv2.imshow("Thresh", imgThresh)
    # cv2.imshow("Warp", imgWarp)
    # cv2.imshow("Warp Points", imgWarpPoints)
    # cv2.imshow("Histogram", imgHist)
    # cv2.imshow("Merged_abhinav", merged_abhi)

    # NORMALIZATION
    curve = curve / 100
    if curve > 1:
        curve == 1
    if curve < -1:
        curve == -1

    return curve


if __name__ == "__main__":
    cap = cv2.VideoCapture("challenge_video.mp4")

    initialTrackVals = [214, 20, 121, 221]
    utils.initializeTrackVals(initialTrackVals)

    frameCounter = 0

    while cap.isOpened():
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, frame = cap.read()
        frame = cv2.resize(frame, (480, 240))

        timer = cv2.getTickCount()

        curve = getLaneCurve(frame, display=2)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        print(f"Curve: {curve}, FPS: {fps}")

        cv2.imshow("Input Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
