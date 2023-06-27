import cv2
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def track_basketball(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Read the first frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error reading frame")
        return

    # Create a window to display the video
    cv2.namedWindow("Basketball Tracking")

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the parameters for the Shi-Tomasi corner detection
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # Find the corners in the first frame
    corners = cv2.goodFeaturesToTrack(gray_frame, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)

    # Half court boundary coordinates
    boundary_x = frame.shape[1] // 2

    # Initialize the indicator variable and previous ball location
    ball_location = None
    prev_ball_location = None

    # Initialize segment start time and flag
    segment_start_time = 0
    segment_flag = False

    while True:
        # Read the next frame
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow using Lucas-Kanade method
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=gray_frame,
            nextImg=gray_prev,
            prevPts=corners,
            nextPts=None
        )

        # Select good points
        good_new = new_corners[status == 1]
        good_old = corners[status == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

            # Check if the ball is on the right or left portion of the half court
            if a > boundary_x:
                ball_location = "Right"
            else:
                ball_location = "Left"

        # Display the ball location
        cv2.putText(frame, f"Ball Location: {ball_location}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check if the ball location has changed
        if ball_location != prev_ball_location:
            if not segment_flag:
                # Start a new segment
                segment_start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                segment_flag = True
            else:
                # End the current segment and save it
                segment_end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                segment_file_path = f"segment_{segment_start_time}_{segment_end_time}.mp4"
                ffmpeg_extract_subclip(video_path, segment_start_time, segment_end_time, targetname=segment_file_path)
                print(f"Segment saved: {segment_file_path}")
                segment_start_time = segment_end_time
        prev_ball_location = ball_location

        # Display the resulting frame
        img = cv2.add(frame, mask)
        cv2.imshow("Basketball Tracking", img)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update previous frame and points
        gray_prev = gray_frame.copy()
        corners = good_new.reshape(-1, 1, 2)

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


# Specify the path to your video file
video_path = "/Users/mainframe/Desktop/defense/video/sv1.mp4"

# Call the function to track the basketball
track_basketball(video_path)

