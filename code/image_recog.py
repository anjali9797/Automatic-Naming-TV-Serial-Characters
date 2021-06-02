# import the libraries
import os
import face_recognition

image_to_be_matched = face_recognition.load_image_file('pic1.jpg')

image_to_be_matched_encoded = face_recognition.face_encodings(
    image_to_be_matched)[0]

print(image_to_be_matched_encoded)

encodings=face_recognition.face_encodings(image_to_be_matched)
print(len(encodings))