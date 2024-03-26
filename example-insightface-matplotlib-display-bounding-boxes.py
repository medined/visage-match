#!/bin/env python

import cv2
import insightface
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load the known image
known_image = cv2.imread("known-faces-01.jpg")

# Convert the image from BGR (OpenCV format) to RGB (for insightface and matplotlib)
known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)

# Initialize insightface's face detector
detector = insightface.app.FaceAnalysis()

# Prepare the context and load the model
detector.prepare(ctx_id=-1, det_size=(640, 640))

# Perform face detection
faces = detector.get(known_image_rgb)

# Check if faces are found
if not faces:
    print("No faces found in the known image")
    exit(1)

# Convert the image to a matplotlib-compatible format
plt.imshow(known_image_rgb)

# Create a matplotlib figure and axis for plotting
ax = plt.gca()

size_threshold = 75

# Loop through each face found in the image
for face in faces:
    bbox = face.bbox.astype(int)
    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]

    if w > size_threshold and h > size_threshold:
        print(x, y, w, h)
        # Create a rectangle patch for each face with a border, but no fill
        rect = Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)

        # Add the rectangle patch to the plot
        ax.add_patch(rect)

# Display the image with the bounding boxes
plt.axis("off")  # Hide the axis labels and ticks
plt.show()
