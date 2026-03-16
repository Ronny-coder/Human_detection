import cv2 as cv
import mediapipe as mp
import time
import math

class PostEstimator():

    # Initialize the Pose model
    def __init__(self, mode=False, upBody =False, smooth = True,
                 detectionCon = 0.5, trackCon = 0.5):
        
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        # Initialize the Pose model
        # Use static_image_mode for images, not for video
        # i get it from chat gpt because i was getting error in video
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    # Find pose landmarks
    def findPose(self, img, draw=True):
        
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)
        
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw = True):

        self.lmList = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, lm)
                self.lmList.append([id, cx, cy])
                cv.circle(img, (cx,cy), 5, (255,0,0), cv.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw = True):

        # Get the landmarks
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        angle = abs(angle*180.0/math.pi)
        #print(angle)
        if angle < 0:
            angle += 360
        

        #draw the angle
        if draw:

            # Draw the lines and circles required for the angle
            cv.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
            cv.line(img, (x3,y3), (x2,y2), (255,0,255), 3)
            cv.circle(img, (x1,y1), 10, (0,0,255), cv.FILLED)
            cv.circle(img, (x1,y1), 15, (0,0,255), 2)
            cv.circle(img, (x2,y2), 10, (0,0,255), cv.FILLED)
            cv.circle(img, (x2,y2), 15, (0,0,255), 2)
            cv.circle(img, (x3,y3), 10, (0,0,255), cv.FILLED)
            cv.circle(img, (x3,y3), 15, (0,0,255), 2)

            # Put the angle text
            cv.putText(img, str(int(angle)), (x2-50,y2+50), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        return angle
        
            


def main():
    cap = cv.VideoCapture("C:/Users/Ranit/OneDrive/Desktop/1.mp4")

    #intialize the pTime    
    pTime = 0
    estimator = PostEstimator()
    while True:
        success, img = cap.read()
        img = estimator.findPose(img,draw=True)
        lmList = estimator.getPosition(img, draw=True )
        if len(lmList) != 0:
            print(lmList[26]) # print the position of landmark 26
            cv.circle(img, (lmList[26][1], lmList[26][2]), 10, (0,128,255), cv.FILLED)

        cTime = time.time()
        fps = 1/(cTime - pTime) if cTime != pTime else 0
        pTime = cTime


        cv.putText( img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        cv.imshow("Image", img)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    main()