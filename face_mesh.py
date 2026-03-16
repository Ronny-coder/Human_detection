import cv2 as cv
import mediapipe as mp
import time 

#cap = cv.VideoCapture("C:/Users/Ranit/OneDrive/Desktop/5.mp4")
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, 
                                  drawSpec, drawSpec
                                 )
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm. y * ih)
                print(id, x, y)

    cTime = time.time()
    fps = 1/(cTime - pTime) if pTime != 0 else 0
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (20,70), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0),3)
    cv.imshow("Image", img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break