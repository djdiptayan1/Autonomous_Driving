import re
import cv2
import numpy as np


def thresholding(image):
    # red
    lower_white_hsv = np.array([100, 65, 35])
    upper_white_hsv = np.array([179, 188, 255])

    lower_white_hsl = np.array([114, 39, 38])
    upper_white_hsl = np.array([179, 255, 118])

    lower_white_rgb = np.array([0, 25, 22])
    upper_white_rgb = np.array([255, 73, 255])

    lower_white_lab = np.array([0, 140, 49])
    upper_white_lab = np.array([112, 255, 255])

    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # white
    # lower_white = np.array([0, 0, 200])
    # upper_white = np.array([180, 30, 255])

    # maskWhite = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)

    # return maskWhite

    rgb = image
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Define filter thresholds for each color space
    lower_rgb = np.array(lower_white_rgb, dtype=np.uint8)
    upper_rgb = np.array(upper_white_rgb, dtype=np.uint8)

    lower_hls = np.array(lower_white_hsl, dtype=np.uint8)
    upper_hls = np.array(upper_white_hsl, dtype=np.uint8)

    lower_hsv = np.array(lower_white_hsv, dtype=np.uint8)
    upper_hsv = np.array(upper_white_hsv, dtype=np.uint8)

    lower_lab = np.array(lower_white_lab, dtype=np.uint8)
    upper_lab = np.array(upper_white_lab, dtype=np.uint8)

    # Apply the filters
    mask_rgb = cv2.inRange(rgb, lower_rgb, upper_rgb)
    mask_hls = cv2.inRange(hls, lower_hls, upper_hls)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_lab = cv2.inRange(lab, lower_lab, upper_lab)

    # Combine the masks with RGB, HSL, HSV, and LAB
    # combined_mask = cv2.bitwise_or(
    #     mask_rgb, cv2.bitwise_or(mask_hls, cv2.bitwise_or(mask_hsv, mask_lab))
    # )

    # Combine the masks with HSV, HSL, and LAB
    combined_mask = cv2.bitwise_or(mask_hls, cv2.bitwise_or(mask_hsv, mask_lab))

    return combined_mask


def warpImg(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp


def nothing(a):
    pass


def initializeTrackVals(intialTracbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar(
        "Width Bottom", "Trackbars", intialTracbarVals[2], wT // 2, nothing
    )
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)


def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.array(
        [
            (widthTop, heightTop),
            (wT - widthTop, heightTop),
            (widthBottom, heightBottom),
            (wT - widthBottom, heightBottom),
        ],
        dtype=np.float32,
    )
    return points


def drawPoints(image, points):
    for x in range(0, 4):
        cv2.circle(
            image, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED
        )
    return image


def getHistogram(image, minPer=0.1, display=True, region=1):
    if region == 1:
        histValue = np.sum(image, axis=1)
    else:
        histValue = np.sum(image[image.shape[0] // region :, :], axis=0)

    # print(hist)
    maxVal = np.max(histValue)
    minVal = minPer * maxVal

    indexArray = np.where(histValue >= minVal)
    basePoint = int(np.average(indexArray))

    # print(basePoint)

    if display:
        imgHist = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

        for x, intensity in enumerate(histValue):
            cv2.line(
                imgHist,
                (x, image.shape[0]),
                (x, image.shape[0] - int(intensity // 255 // region)),
                (255, 0, 255),
                1,
            )
            cv2.circle(
                imgHist,
                (basePoint, image.shape[0]),
                10,
                (0, 255, 255),
                cv2.FILLED,
            )
        return basePoint, imgHist
    return basePoint


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale
                    )
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y],
                        (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                        None,
                        scale,
                        scale,
                    )
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x],
                    (imgArray[0].shape[1], imgArray[0].shape[0]),
                    None,
                    scale,
                    scale,
                )
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
