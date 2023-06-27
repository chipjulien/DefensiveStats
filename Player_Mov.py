import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture("basketball_game.mp4")

# Create a multi-object tracker using the CSRT algorithm
tracker = cv2.MultiTracker_create()

# Read the first frame
ret, frame = cap.read()

# Define the ROI for each player
roi1 = cv2.selectROI(frame, False)
roi2 = cv2.selectROI(frame, False)
roi3 = cv2.selectROI(frame, False)
roi4 = cv2.selectROI(frame, False)
roi5 = cv2.selectROI(frame, False)
roi6 = cv2.selectROI(frame, False)
roi7 = cv2.selectROI(frame, False)
roi8 = cv2.selectROI(frame, False)
roi9 = cv2.selectROI(frame, False)
roi10 = cv2.selectROI(frame, False)

# Add the ROIs to the tracker
tracker.add(cv2.TrackerCSRT_create(), frame, roi1)
tracker.add(cv2.TrackerCSRT_create(), frame, roi2)
tracker.add(cv2.TrackerCSRT_create(), frame, roi3)
tracker.add(cv2.TrackerCSRT_create(), frame, roi4)
tracker.add(cv2.TrackerCSRT_create(), frame, roi5)
tracker.add(cv2.TrackerCSRT_create(), frame, roi6)
tracker.add(cv2.TrackerCSRT_create(), frame, roi7)
tracker.add(cv2.TrackerCSRT_create(), frame, roi8)
tracker.add(cv2.TrackerCSRT_create(), frame, roi9)
tracker.add(cv2.TrackerCSRT_create(), frame, roi10)

# Loop over each frame in the video
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Break the loop if the frame is not available
    if not ret:
        break

    # Update the tracker and get the updated ROIs
    success, boxes = tracker.update(frame)

    # Draw rectangles around the tracked objects
    for box in boxes:
        x, y, w, h = [int(i) for i in box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

    # Show the resulting frame
    cv2.imshow("Basketball Game", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
