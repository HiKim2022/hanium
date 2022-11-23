from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pyfirmata

# function
def cal_distance(ptA, ptB):
    return np.linalg.norm(ptA - ptB)


def cal_EAR(eye);
  D1 = cal_distance(eye[1], eye[5])
  D2 = cal_distance(eye[2], eye[4])
  D3 = cal_distance(eye[0], eye[3])
  EAR = (D1+D2) / (2.0*D3)
  return EAR


# construct 
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=int, default=0, help="boolean used to indicate if TraffHat should be used")
args = vars(ap.parse_args())
board = pyfirmata.Arduino('/dev/ttyACM0')
pin9=board.get_pin('d:10:o')


if args["alarm"] > 0:
    from gpiozero import TrafficHat

    th = TrafficHat()
    print("[INFO] using TrafficHat alarm...")


#threadshold and init    
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 16
COUNTER = 0
ALARM_ON = False


print("loading facial landmark predictor...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print(" starting video stream thread...")

vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)
# ----------------------------------------------------------------------------


while (cap.isOpened()):
   #stream
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

   #calculate EAR
    for (x, y, w, h) in rects:
        
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

       
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = cal_EAR(leftEye)
        rightEAR = cal_EAR(rightEye)

       
        ear = (leftEAR + rightEAR) / 2.0

       
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        
        # detection drowsiness
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
               
                if not ALARM_ON:
                    ALARM_ON = True
                    pin9.write(1)

                   
                    if args["alarm"] > 0:
                        th.buzzer.blink(0.1, 0.1, 10, background=True)

           
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

     
        else:
            COUNTER = 0
            ALARM_ON = False
            pin9.write(0)

        
        cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    #exit
    if key == ord("q"):
        break

  
