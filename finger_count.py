import cv2 as cv
import time
import os
import hand_tracking_module as htm

wCam, hCam = 640, 480

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0


mylist = os.listdir("Finger_tips")
print(mylist)
overlayList = []
for imPath in mylist:
    image = cv.imread(f'Finger_tips/{imPath}')
    overlayList.append(image)

print(len(overlayList))

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:

    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    #print(lmlist)

    if len(lmlist) != 0:
        finger = []
        #thumb
        if lmlist[tipIds[0]][1] > lmlist[tipIds[0]-1][1]:
            finger.append(1)
                
        else:
            finger.append(0)
        
        # four fingers
        for id in range(1,5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                finger.append(1)
                
            else:
                finger.append(0)

        #print(finger)
            totalFingers = finger.count(1)
            print(totalFingers)
            

        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        rect_start_y = h + 20
        rect_end_y = rect_start_y + 150

        cv.rectangle(img, (20, rect_start_y), (170, rect_end_y), (0,255,0), cv.FILLED)

        cv.putText(img, str(totalFingers), (45, rect_start_y + 120),
           cv.FONT_HERSHEY_PLAIN,
           10, (255,0,0), 25)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime 

    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN,
                3, (255,255,0), 3)

    cv.imshow("Image", img)

    if cv.waitKey(1) & 0xFF == ord('q') :
        break
