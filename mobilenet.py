import numpy as np
import cv2
from tracker import *
from imutils.video import VideoStream
from imutils.video import FPS
import dlib

tracker = EuclideanDistTracker()

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"
INP_VIDEO_PATH = 'video/video.mp4'
OUT_VIDEO_PATH = 'output/result.mp4'
GPU_SUPPORT = 0
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",  "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def convertMillis(millis):
     seconds=(millis/1000)%60
     minutes=(millis/(1000*60))%60
     hours=(millis/(1000*60*60))%24
     return "{0}:{1}:{2}".format(int(hours),int(minutes),int(seconds))


net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    
cap = cv2.VideoCapture(INP_VIDEO_PATH)

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    '''fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter('output/result1.mp4', fourcc ,10,(w,h), True)'''
    millis = cap.get(cv2.CAP_PROP_POS_MSEC)
    tracking = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}".format(CLASSES[idx])
            tracking.append([startX,startY,endX,endY])

    boxes_ids = tracker.update(tracking)
    
    for box_id in boxes_ids:
        startX,startY,endX,endY,id = box_id
        cv2.rectangle(frame, (startX, startY), (endX, endY),    COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
    print(label + " at " + str(convertMillis(millis)))
    cv2.imshow("Image", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()