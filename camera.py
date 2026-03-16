import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # try 0, 1, 2...
if not cap.isOpened():
    print("❌ Camera not accessible. Check privacy settings or try another index.")
else:
    print("✅ Camera is accessible!")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow("Camera Test", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
