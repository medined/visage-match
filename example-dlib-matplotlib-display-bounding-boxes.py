#!/bin/env python

import cv2
import dlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load the known image
known_image = cv2.imread("known-faces-02.jpg")

# Convert the image from BGR (OpenCV format) to RGB (for dlib and matplotlib)
known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)

# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Perform face detection
detections = detector(
    known_image_rgb, 1
)  # The 1 here indicates that we should upsample the image 1 time. This will make everything bigger and allow us to detect more faces.

# Check if faces are found
if len(detections) == 0:
    print("No faces found in the known image")
    exit(1)

# Convert the image to a matplotlib-compatible format
plt.imshow(known_image_rgb)

# Create a matplotlib figure and axis for plotting
ax = plt.gca()

size_threshold = 75

# Loop through each face found in the image
for detection in detections:
    x, y, w, h = (
        detection.left(),
        detection.top(),
        detection.width(),
        detection.height(),
    )

    if w > size_threshold and h > size_threshold:
        print(x, y, w, h)
        # Create a rectangle patch for each face with a border, but no fill
        rect = Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)

        # Add the rectangle patch to the plot
        ax.add_patch(rect)

# Display the image with the bounding boxes
plt.axis("off")  # Hide the axis labels and ticks
plt.show()
