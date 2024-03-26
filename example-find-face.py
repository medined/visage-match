#!/bin/env python

import face_recognition
from icecream import ic

# Load the known image

known_image = face_recognition.load_image_file("known-faces-01-min.jpg")
known_image_encoding = face_recognition.face_encodings(known_image, model="cnn")
if not known_image_encoding:
    print("No faces found in the known image")
    exit(1)

print(f"Faces Found: {len(known_image_encoding)}")
