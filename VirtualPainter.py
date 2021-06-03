import cv2, time
import mediapipe as mp
import HandModule as hm
import numpy as np

#####################
wcam, hcam = 640, 480
#####################

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
ptime = 0

detector = hm.HandDetector(detectionCon=0.85)
SelectedColor = (255, 255, 255)
tipIds = [8, 12]

imgCanvas = np.zeros((480, 640, 3), np.uint8)

xp, yp = 0, 0
brushThickness = 10 
eraserThickness = 100

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Detecting and Finding Hand Landmarks
    img = detector.FindHands(img)
    lmlist = detector.FindPos(img, draw=False)

    # Check which fingers are up
    if len(lmlist) != 0:
        SelectionMode, DrawingMode = detector.FingersUp(img, lmlist, tipIds)

    # If Selection Mode - Two Finger are up
        if SelectionMode:
            xp, yp = 0, 0
            xt1, yt1 = lmlist[tipIds[0]][1], lmlist[tipIds[0]][2]
            xt2, yt2 = lmlist[tipIds[1]][1], lmlist[tipIds[1]][2]

            if xt1 in range(wcam//5, (wcam//5)*2+1) and yt1 in range(0, wcam//10):
                SelectedColor = (0, 255, 0)      # GREEN
            if xt1 in range(wcam//5*2, (wcam//5)*3+1) and yt1 in range(0, wcam//10):
                SelectedColor = (0, 0, 255)     # BLUE
            if xt1 in range(wcam//5*3, (wcam//5)*4+1) and yt1 in range(0, wcam//10):
                SelectedColor = (255, 0, 0)     # RED
            if xt1 in range(wcam//5*4, (wcam//5)*5+1) and yt1 in range(0, wcam//10):
                SelectedColor = (0, 0, 0)     # WHITE
            cv2.rectangle(img, (xt1, yt1-25), (xt2, yt2+25), SelectedColor, cv2.FILLED)
        cv2.putText(img, 'Color Selected', (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, SelectedColor, 2)


    # If Drawing Mode - One Finger is up
        if DrawingMode:
            x1, y1 = lmlist[tipIds[0]][1], lmlist[tipIds[0]][2]
            cv2.circle(img, (x1, y1), 15, SelectedColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if SelectedColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), SelectedColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), SelectedColor, eraserThickness)

            cv2.line(img, (xp, yp), (x1, y1), SelectedColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), SelectedColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    cv2.rectangle(img, (wcam//5, 0), ((wcam//5)*2, wcam//10), (0, 255, 0), cv2.FILLED) # GREEN
    cv2.rectangle(img, ((wcam//5)*2, 0), ((wcam//5)*3, wcam//10), (0, 0, 255), cv2.FILLED) # BLUE
    cv2.rectangle(img, ((wcam//5)*3, 0), ((wcam//5)*4, wcam//10), (255, 0, 0), cv2.FILLED) # RED
    cv2.rectangle(img, ((wcam//5)*4, 0), ((wcam//5)*5, wcam//10), (255, 255, 255), cv2.FILLED) # WHITE

    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Painter", img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
