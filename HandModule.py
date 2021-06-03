import mediapipe as mp
import cv2


class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def FindHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def FindPos(self, img, handnumber = 0, draw = True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handnumber]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
        return lmList

    def FingersUp(self, img, lmlist, tipIds):
        fingers = []
        SelectionMode, DrawingMode = 0, 0
        for id in range(0, 2):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else: fingers.append(0)

        if sum(fingers) >= 2:
            SelectionMode = 1
            DrawingMode = 0
            cv2.putText(img, 'SelectionMode', (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        elif sum(fingers) == 1:
            DrawingMode = 1
            SelectionMode = 0
            cv2.putText(img, 'DrawingMode', (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        else:
            cv2.putText(img, 'Cannot Draw or Select', (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        return SelectionMode, DrawingMode