import cv2 as cv
import mediapipe as mp
import time 

class handDetector():
    # Initialize the Pose model
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=1,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.result = None

    # Find hand landmarks
    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    
    # Get the position of landmarks
    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        if self.result and self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])

                if draw :  # Example: draw only on index fingertip
                    cv.circle(img, (cx, cy), 6, (255,255,0), cv.FILLED)
        
        return lmlist


def main():
    pTime = 0
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)  
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        # draw hand landmarks or not based on user preference
        img = detector.findHands(img,draw=True)
        lmlist = detector.findPosition(img,draw=False)

        if len(lmlist) != 0:
            print(lmlist[12])  
        
        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime 

        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN,
                3, (0,255,0), 3)
        

        cv.imshow("Image", img)

        # exit on 'q' or window close
        if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()