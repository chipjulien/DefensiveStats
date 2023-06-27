import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Get the first frame
ret, frame = cap.read()

# Convert the frame to grayscale
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Define the region of interest (ROI) where the basketball is expected to be
roi = cv2.selectROI(frame, False)

# Create a mask for the ROI
mask = np.zeros_like(frame)
mask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 1

# Define the termination criteria for the Lucas-Kanade optical flow algorithm
lk_params = dict(winSize=(15,15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize the previous points
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **lk_params)

# Loop over each frame in the video
while True:
    # Get the next frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the optical flow
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    # Select the good points
    good_pts = prev_pts[status == 1]
    next_good_pts = next_pts[status == 1]

    # Draw the flow vectors
    for i, (pt, next_pt) in enumerate(zip(good_pts, next_good_pts)):
        x, y = pt.ravel()
        x_next, y_next = next_pt.ravel()
        cv2.line(frame, (x,y), (x_next, y_next), (0,255,0), 2)

    # Show the resulting frame
    cv2.imshow("Basketball Movement", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the previous frame and points
    prev_gray = gray
    prev_pts = good_pts

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
