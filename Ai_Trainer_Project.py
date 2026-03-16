import cv2 as cv
import numpy as np 
import time 
import post_estimation_module as pm

cap = cv.VideoCapture("C:/Users/Ranit/OneDrive/Desktop/6.mp4")

detector = pm.PostEstimator()

count = 0
dir = 0
pTime = 0

while True:
    success, img = cap.read()
    img = cv.resize(img, (1280, 720))

    img = detector.findPose(img, False)
    lmList = detector.getPosition(img, draw=False)
    #print(lmList)

    if len(lmList) != 0:
        # left arm
        angle = detector.findAngle(img, 11, 13, 15)
        # right arm help
        #detector.findAngle(img, 12, 14, 16)

        per = np.interp(angle, (40, 160), (0, 100))
        #bar = np.interp(lmList[14][2], (50, 200), (650, 100))
        #print(angle, per)

        #per = np.interp(angle, (210, 310 ), (0, 100))
        ber = np.interp(angle, (210, 310), (650, 100))
        #print(angle, per)




        # check for the dumbbell curls
        if per >= 95 and dir == 0:
            count += 0.5
            dir = 1

        if per <= 5 and dir == 1:
            count += 0.5
            dir = 0

        print(count)

        #cv.rectangle(img, (1100,450), (1280,720), (0,255,0), cv.FILLED)
        #cv.rectangle(img, (1100, int(ber)), (1280, 650), (255,0,0), cv.FILLED)
        #cv.putText(img, f'{int(per)} %', (1100, 400), cv.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)

        cv.rectangle(img, (0,450), (300,720), (0,255,0), cv.FILLED)
        cv.putText(img, f' {int(count)}', (35,670), cv.FONT_HERSHEY_PLAIN, 15, (255,0,0), 25)

    cTime = time.time()
    fps = 1/(cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    cv.putText(img, f'FPS: {int(fps)}', (50,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break