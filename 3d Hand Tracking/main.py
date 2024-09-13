import cv2
from cvzone.HandTrackingModule import HandDetector
import socket

# Parameters
width, height = 1280, 720

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detector = HandDetector(maxHands=2, detectionCon=0.8)  # 최대 두 손 인식

# Communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

while True:
    # Get the frame from the webcam
    success, img = cap.read()

    # Flip the image horizontally (좌우 반전)
    img = cv2.flip(img, 1)

    # Hands
    hands, img = detector.findHands(img)

    data = []
    # Landmark values - (x,y,z) * 21 * 2 (for two hands)
    if hands:
        for hand in hands:
            # Get the landmark list for each hand
            lmList = hand['lmList']
            print(lmList)
            for lm in lmList:
                data.extend([lm[0], height - lm[1], lm[2]])
        print(data)
        sock.sendto(str.encode(str(data)), serverAddressPort)

    img = cv2.resize(img, (0, 0), None, 1, 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
