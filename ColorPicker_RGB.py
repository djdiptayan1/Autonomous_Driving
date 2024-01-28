import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("LineVid.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

# Function to update the color picker values
def update_color_picker():
    r_min = cv2.getTrackbarPos("Red Min", "RGB")
    r_max = cv2.getTrackbarPos("Red Max", "RGB")
    g_min = cv2.getTrackbarPos("Green Min", "RGB")
    g_max = cv2.getTrackbarPos("Green Max", "RGB")
    b_min = cv2.getTrackbarPos("Blue Min", "RGB")
    b_max = cv2.getTrackbarPos("Blue Max", "RGB")
    lower = np.array([b_min, g_min, r_min])
    upper = np.array([b_max, g_max, r_max])
    return lower, upper

# Create the trackbars
cv2.namedWindow("RGB")
cv2.resizeWindow("RGB", 640, 480)
cv2.createTrackbar("Red Min", "RGB", 0, 255, empty)
cv2.createTrackbar("Red Max", "RGB", 255, 255, empty)
cv2.createTrackbar("Green Min", "RGB", 0, 255, empty)
cv2.createTrackbar("Green Max", "RGB", 255, 255, empty)
cv2.createTrackbar("Blue Min", "RGB", 0, 255, empty)
cv2.createTrackbar("Blue Max", "RGB", 255, 255, empty)

frameCounter = 0

while True:
    frameCounter += 1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    _, img = cap.read()

    lower, upper = update_color_picker()

    mask = cv2.inRange(img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])

    # Display RGB values
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

    cv2.imshow("RGB Color Picker", hStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
