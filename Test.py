import cv2
import numpy as np
import matplotlib.pyplot as plt

curveList = []
avgVal = 10

cap = cv2.VideoCapture("Test_Vid.mp4")


def vidResize(frame):
    frame = cv2.resize(frame, (854, 480))  # Resize frame to 480P 16:9 ratio
    return frame


def thresholding(img):
    img = cv2.resize(
        img, (427, 240)
    )  # Resize img for easier filter application and faster processing

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower_white_hsv = np.array([0, 24, 144])
    upper_white_hsv = np.array([112, 99, 216])

    lower_white_hsl = np.array([0, 45, 18])
    upper_white_hsl = np.array([131, 255, 135])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    maskWhite_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
    maskWhite_hsl = cv2.inRange(hsl, lower_white_hsl, upper_white_hsl)

    maskWhite = cv2.bitwise_or(maskWhite_hsv, maskWhite_hsl)

    return maskWhite_hsv, maskWhite_hsl, maskWhite


def perspective_transform(frame):
    # Plotting four circles on the video of the object you want to see the transformation of.
    cv2.circle(frame, (114, 151), 5, (0, 0, 255), -1)
    cv2.circle(frame, (605, 89), 5, (0, 0, 255), -1)
    cv2.circle(frame, (72, 420), 5, (0, 0, 255), -1)
    cv2.circle(frame, (637, 420), 5, (0, 0, 255), -1)

    # Selecting all the above four points in an array
    imgPts = np.float32([[114, 151], [605, 89], [72, 420], [637, 420]])

    # Selecting four points in an array for the destination video( the one you want to see as your output)
    objPoints = np.float32([[0, 0], [420, 0], [0, 637], [420, 637]])

    # Apply perspective transformation function of openCV2. This function will return the matrix which you can feed into warpPerspective function to get the warped image.
    matrix = cv2.getPerspectiveTransform(imgPts, objPoints)
    result = cv2.warpPerspective(frame, matrix, (400, 600))

    return result


def getHistogram(img):
    pass


def getLaneCurve(image):
    img = image.copy()
    # Warp the image to get a bird's eye view
    imgWarp = perspective_transform(img)
    # Threshold the image
    imgThresh = thresholding(imgWarp)[2]

    # GET HISTOGRAM

    return imgThresh


frameCounter = 0
if __name__ == "__main__":
    while cap.isOpened():
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, frame = cap.read()
        frame = vidResize(frame)  # ORIGINAL INPUT VIDEO

        curve = getLaneCurve(frame)
        warp = perspective_transform(frame)
        HSV_mask, HSL_mask, result = thresholding(warp)

        print(frame.shape, HSV_mask.shape, HSL_mask.shape, curve.shape)

        # Create headings for each frame
        heading1 = "HSV Filter"
        heading2 = "HSL Filter"
        heading3 = "Final Result"

        # Put text on each frame
        cv2.putText(
            HSV_mask,
            heading1,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            HSL_mask,
            heading2,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            heading3,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Stack FILTER frames horizontally
        hstack = np.hstack([HSV_mask, HSL_mask, result])

        cv2.imshow("Input Video", frame)
        cv2.imshow("Warp", warp)
        cv2.imshow("Filters", hstack)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
