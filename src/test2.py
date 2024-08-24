from random import random

import cv2

from src.hrFaceDetection import MIN_FACE_SIZE, distance, ADD_BOX_ERROR, BOX_ERROR_MAX, getROI, CASCADE_PATH


def getBestROI(frame, faceCascade, previousFaceBox):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
                                         flags=cv2.CASCADE_SCALE_IMAGE)  # flags=cv2.cv2.CV_HAAR_SCALE_IMAGE
    roi = None
    faceBox = None

    # If no face detected, use ROI from previous frame
    if len(faces) == 0:
        print("called1")
        faceBox = previousFaceBox

    # if many faces detected, use one closest to that from previous frame
    elif len(faces) > 1:
        # print("called2")
        if previousFaceBox is not None:
            # Find closest
            minDist = float("inf")
            for face in faces:
                if distance(previousFaceBox, face) < minDist:
                    faceBox = face
        else:
            # Chooses largest box by area (most likely to be true face)
            print("called3")
            maxArea = 0
            for face in faces:
                if (face[2] * face[3]) > maxArea:
                    faceBox = face

    # If only one face dectected, use it!
    else:
        faceBox = faces[0]

    if faceBox is not None:
        if ADD_BOX_ERROR:
            noise = []
            for i in range(4):
                noise.append(random.uniform(-BOX_ERROR_MAX, BOX_ERROR_MAX))
            (x, y, w, h) = faceBox
            x1 = x + int(noise[0] * w)
            y1 = y + int(noise[1] * h)
            x2 = x + w + int(noise[2] * w)
            y2 = y + h + int(noise[3] * h)
            faceBox = (x1, y1, x2 - x1, y2 - y1)

        # Show rectangle
        # (x, y, w, h) = faceBox
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        roi = getROI(frame, faceBox)

    return faceBox, roi


video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_PATH)

colorSig = []  # Will store the average RGB color values in each frame's ROI
heartRates = []  # Will store the heart rate calculated every 1 second
previousFaceBox = None
while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        break

    previousFaceBox, roi = getBestROI(frame, faceCascade, previousFaceBox)

    # cv2.imshow('ROI', roi)
    print(roi)
    cv2.waitKey(1)