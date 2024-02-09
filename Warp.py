# Importing Libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Capturing video
cap = cv2.VideoCapture("Test_Vid.mp4")
while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (854, 480))  # Resize frame to 480P 16:9 ratio
    # Plotting four circles on the video of the object you want to        see the transformation of.
    cv2.circle(frame, (114, 151), 5, (0, 0, 255), -1)
    cv2.circle(frame, (605, 89), 5, (0, 0, 255), -1)
    cv2.circle(frame, (72, 420), 5, (0, 0, 255), -1)
    cv2.circle(frame, (637, 420), 5, (0, 0, 255), -1)
    # selecting all the above four points in an array
    imgPts = np.float32([[114, 151], [605, 89], [72, 420], [637, 420]])

    # selecting four points in an array for the destination video( the one you want to see as your output)
    objPoints = np.float32([[0, 0], [420, 0], [0, 637], [420, 637]])
    # Apply perspective transformation function of openCV2. This function will return the matrix which you can feed into warpPerspective function to get the warped image.
    matrix = cv2.getPerspectiveTransform(imgPts, objPoints)
    result = cv2.warpPerspective(frame, matrix, (400, 600))
    # Now Plotting both the videos(original, warped video)using matplotlib
    cv2.imshow("frame", frame)
    cv2.imshow("Perspective Transformation", result)
    key = cv2.waitKey(1)
    plt.show()
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
