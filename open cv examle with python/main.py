import cv2
import numpy as np

cap = cv2.VideoCapture("turuncu.mp4") # upload your video
fgbg = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read() # Get a frame from the video

    if ret:
        # Convert image to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Identify and mask the range of the orange color
        lower_orange = np.array([0, 70, 50])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Identify and mask the range of the blue color
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Remove the background
        fgmask = fgbg.apply(frame)

        # Find the contours for the mask of the orange color
        contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find the contours for the mask of the blue color
        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # show frame
        cv2.imshow("Frame", frame)


    # Exit when pressing "q" key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break