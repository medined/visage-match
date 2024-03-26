#!/bin/env python

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load the known image
known_image = cv2.imread("known-faces-01.jpg")

# Convert the image from BGR (OpenCV format) to RGB (for matplotlib)
known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)

# Load OpenCV's pre-trained Haar Cascade face detector model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Perform face detection
faces = face_cascade.detectMultiScale(
    known_image_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
)

# Check if faces are found
if len(faces) == 0:
    print("No faces found in the known image")
    exit(1)

# Convert the image to a matplotlib-compatible format
plt.imshow(known_image_rgb)

# Create a matplotlib figure and axis for plotting
ax = plt.gca()

size_threshold = 250

# Loop through each face found in the image
for x, y, w, h in faces:
    if w > size_threshold and h > size_threshold:
        print(x, y, w, h)
        # Create a rectangle patch for each face with a border, but no fill
        rect = Rectangle((x, y), w, h, fill=False, color="red", linewidth=2)

        # Add the rectangle patch to the plot
        ax.add_patch(rect)

# Display the image with the bounding boxes
plt.axis("off")  # Hide the axis labels and ticks
plt.show()
