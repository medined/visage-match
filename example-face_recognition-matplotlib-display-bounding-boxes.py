#!/bin/env python

import face_recognition
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

known_image = face_recognition.load_image_file("known-faces-01.jpg")

# Find all face locations in the image
face_locations = face_recognition.face_locations(known_image, model="hog")
if not face_locations:
    print("No faces found in the known image")
    exit(1)

face_encodings = face_recognition.face_encodings(
    known_image, known_face_locations=face_locations
)

# Convert the image to a matplotlib-compatible format
plt.imshow(known_image)

# Create a matplotlib figure and axis for plotting
ax = plt.gca()

# Define the size threshold (minimum width and height) for the bounding box
min_width, min_height = 50, 50

# Loop through each face found in the image
for face_location in face_locations:
    # Each face_location contains the positions (top, right, bottom, left)
    top, right, bottom, left = face_location
    width, height = right - left, bottom - top

    if width >= min_width and height >= min_height:
        # Create a rectangle patch for each face with a border, but no fill
        rect = Rectangle(
            (left, top), width, height, fill=False, color="red", linewidth=2
        )

        # Add the rectangle patch to the plot
        ax.add_patch(rect)

# Display the image with the bounding boxes
plt.axis("off")  # Hide the axis labels and ticks
plt.show()
