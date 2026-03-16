import cv2 as cv
import time
import numpy as np
import hand_tracking_module as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


wCam, hCam = 640, 480


cap = cv.VideoCapture(0, cv.CAP_DSHOW)  
cap.set(3, wCam)
cap.set(4, hCam)


pTime = 0

detector = htm.handDetector(detectionCon = 0.75)


#devices = AudioUtilities.GetSpeakers()
#interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#volume = cast(interface, POINTER(IAudioEndpointVolume))

#volume = GetMute()
#volume = GetMasterVolumeLevel()
#devices = AudioUtilities.GetSpeakers()
#volume = devices.EndpointVolume
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        #print(lmList[4], lmList[8]) 

        x1, y1, = lmList[4][1], lmList[4][2]
        x2, y2, = lmList[8][1], lmList[8][2]

        cv.circle(img, (x1, y1), 10, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 10, (255, 0, 255), cv.FILLED)
        
        cx,cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2-x1, y2-y1)
        #print(length)

        # Hand range 50 - 300
        # volume range -65 - 0
        vol = np.interp(length, [50,300], [minVol, maxVol])
        print(int(length), vol  )
        volume.SetMasterVolumeLevel(vol, None)
        volBar = np.interp(length, [50,300], [400, 150])
        volPer = np.interp(length, [50,300], [0, 100])

        if length < 50:
            cv.circle(img, (cx, cy,), 15, (0, 255, 0), cv.FILLED)

    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(img, f'Volume: {int(volPer)}%', (40, 450), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Img", img)
    if cv.waitKey(1) & 0xFF == ord('q') :
        break