import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2 as cv
import time
import post_estimation_module as p


def main():
    cap = cv.VideoCapture("C:/Users/Ranit/OneDrive/Desktop/1.mp4")

    #intialize the pTime    
    pTime = 0
    estimator = p.PostEstimator()
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