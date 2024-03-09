import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass


# Function to update the HSL color picker values
def update_color_picker_hsl():
    h_min = cv2.getTrackbarPos("Hue Min", "HSL")
    h_max = cv2.getTrackbarPos("Hue Max", "HSL")
    s_min = cv2.getTrackbarPos("Saturation Min", "HSL")
    s_max = cv2.getTrackbarPos("Saturation Max", "HSL")
    l_min = cv2.getTrackbarPos("Lightness Min", "HSL")
    l_max = cv2.getTrackbarPos("Lightness Max", "HSL")
    lower = np.array([h_min, s_min, l_min])
    upper = np.array([h_max, s_max, l_max])
    return lower, upper


# Create the HSL trackbars
cv2.namedWindow("HSL")
cv2.resizeWindow("HSL", 640, 480)
cv2.createTrackbar("Hue Min", "HSL", 0, 179, empty)
cv2.createTrackbar("Hue Max", "HSL", 179, 179, empty)
cv2.createTrackbar("Saturation Min", "HSL", 0, 255, empty)
cv2.createTrackbar("Saturation Max", "HSL", 255, 255, empty)
cv2.createTrackbar("Lightness Min", "HSL", 0, 255, empty)
cv2.createTrackbar("Lightness Max", "HSL", 255, 255, empty)

frameCounter = 0

while True:
    frameCounter += 1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0

    _, img = cap.read()

    height, width, _ = img.shape
    width //= 2
    left_frame = img[:, :width, :]
    right_frame = img[:, width:, :]

    lower, upper = update_color_picker_hsl()

    img_hsl = cv2.cvtColor(right_frame, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(img_hsl, lower, upper)
    result = cv2.bitwise_and(right_frame, right_frame, mask=mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([right_frame, mask, result])

    # Display HSL values
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

    cv2.imshow("HSL Color Picker", hStack)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
