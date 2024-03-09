import cv2

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Failed to open the camera")
        return

    # Capture and display the camera feeds
    while True:
        # Read a new frame from the camera
        ret, frame = cap.read()

        # Split the stereo frame into left and right images
        height, width, _ = frame.shape
        width //= 2
        left_frame = frame[:, :width, :]
        right_frame = frame[:, width:, :]

        # Display the left and right camera feeds separately
        cv2.imshow("Left Camera Feed", left_frame)
        cv2.imshow("Right Camera Feed", right_frame)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
