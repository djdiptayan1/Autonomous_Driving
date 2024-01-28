import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("LineVid.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass


# Function to update the LAB color picker values
def update_lab_picker():
    l_min = cv2.getTrackbarPos("L Min", "LAB")
    l_max = cv2.getTrackbarPos("L Max", "LAB")
    a_min = cv2.getTrackbarPos("A Min", "LAB")
    a_max = cv2.getTrackbarPos("A Max", "LAB")
    b_min = cv2.getTrackbarPos("B Min", "LAB")
    b_max = cv2.getTrackbarPos("B Max", "LAB")
    lower = np.array([l_min, a_min, b_min])
    upper = np.array([l_max, a_max, b_max])
    return lower, upper


# Create the trackbars for LAB
cv2.namedWindow("LAB")
cv2.resizeWindow("LAB", 640, 480)
cv2.createTrackbar("L Min", "LAB", 0, 255, empty)
cv2.createTrackbar("L Max", "LAB", 255, 255, empty)
cv2.createTrackbar("A Min", "LAB", 0, 255, empty)
cv2.createTrackbar("A Max", "LAB", 255, 255, empty)
cv2.createTrackbar("B Min", "LAB", 0, 255, empty)
cv2.createTrackbar("B Max", "LAB", 255, 255, empty)

frameCounter = 0

while True:
    frameCounter += 1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    _, img = cap.read()

    lower, upper = update_lab_picker()

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])

    # Display LAB values
    cv2.putText(
        hStack,
        f"Lower: {lower}",
        (10, frameHeight - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        hStack,
        f"Upper: {upper}",
        (10, frameHeight - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("LAB Color Picker", hStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
