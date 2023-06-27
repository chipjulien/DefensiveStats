import cv2
import numpy as np

# Load the basketball court image
img = cv2.imread("basketball_court.jpg")

# Define the vertices of the court sections
key_points = np.array([[250, 0], [250, 150], [400, 150], [400, 0],
                       [0, 150], [150, 150], [150, 300], [0, 300],
                       [400, 150], [550, 150], [550, 300], [400, 300],
                       [150, 300], [250, 300], [250, 450], [150, 450],
                       [400, 300], [500, 300], [500, 450], [400, 450]], np.int32)

# Reshape the key points into a form that can be passed to the polylines function
key_points = key_points.reshape((-1, 1, 2))

# Draw the polylines to highlight the court sections
cv2.polylines(img, [key_points], True, (255,0,0), thickness=2)

# Show the resulting image
cv2.imshow("Basketball Court", img)

# Wait until a key is pressed
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()
