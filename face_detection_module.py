import cv2 as cv
import mediapipe as mp 
import time 


class FaceDetector:
    # Initialize the FaceDetector with a minimum detection confidence
    def __init__(self, minDetectionCon=0.75):

        # Store the minimum detection confidence
        # and set up the Mediapipe face detection model
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection 
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection  = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        # Convert the image to RGB as Mediapipe requires RGB images
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        # Initialize a list to hold bounding box information
        bboxs = []

        # If faces are detected, process each detection
        # create a bounding box around each detected face
        # and store its information
        if self.results.detections:
            # Loop through each detected face
            for id,detections in enumerate(self.results.detections):

                # Extract the bounding box coordinates
                bboxC = detections.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin *iw) , int(bboxC.ymin *ih), \
                    int(bboxC.width *iw), int(bboxC.height *ih)
                
                bboxs.append([id, bbox, detections.score])
                img = self.fancyDraw(img, bbox)
                
                # Draw the bounding box and confidence score on the image
                cv.rectangle(img, bbox, (255,0,0),2)
                cv.putText(img, f'ID: {int(detections.score[0] *100)}%',
                            (bbox[0], bbox[1]-10), 
                        cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
                
        return img, bboxs
    
    # function to draw fancy bounding boxes
    def fancyDraw(self, img, bbox, l=30, t=10):

        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv.rectangle(img, bbox, (255, 0, 255), 2)

        # Top Left
        cv.line(img, (x, y), (x+l , y), (255, 0, 255), t)
        cv.line(img, (x, y), (x , y+l), (255, 0, 255), t)

        # Top Right
        cv.line(img, (x1, y), (x1 - l , y), (255, 0, 255), t)
        cv.line(img, (x1, y), (x1 , y + l), (255, 0, 255), t)

        # Bottom Left
        cv.line(img, (x, y1), (x + l , y1), (255, 0, 255), t)
        cv.line(img, (x, y1), (x , y1 - l), (255, 0, 255), t)

        # Bottom Right
        cv.line(img, (x1, y1), (x1 - l , y1), (255, 0, 255), t)
        cv.line(img, (x1, y1), (x1 , y1 - l), (255, 0, 255), t)
        return img
   
def main():

    # this capture the video from the given path
    # and it will read the video frame by frame
    cap = cv.VideoCapture("C:/Users/Ranit/OneDrive/Desktop/5.mp4")
    # initialize the previous time for FPS calculation
    pTime = 0

    detector = FaceDetector()
   
    # main loop to process each frame
    while True:
        success, img = cap.read()
        if not success:
            break

        # detect faces in the frame
        img, bboxs = detector.findFaces(img)
        print(bboxs)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1/(cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        cv.putText( img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv.imshow("Image", img)
        cv.waitKey(10)


if __name__ == "__main__":
    main()