#!/usr/bin/python3
import sys
import numpy as np
import cv2 as cv
import time
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from utils import p2l,gridmap_generator


# Human detector
net = cv.dnn.readNet("dnn_model/yolov4-tiny.weights",'dnn_model/yolov4-tiny.cfg')
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale =1/255) 
nmsThreshold = 0.4
confThreshold = 0.5
classes = []
with open("dnn_model/classes.txt") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
        
#Tracker Algorithm
tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)       

# Source video
cap = cv.VideoCapture(str(sys.argv[1]))
cap.set(cv.CAP_PROP_FRAME_WIDTH,960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,540)
h = 960
w = 540
# Output video
result = cv.VideoWriter(str(sys.argv[2]+".mp4"),cv.VideoWriter_fourcc(*'MJPG'),25, (960,540))

# Grid Map 
cs = 20   # Cell_size
cell_x    = int(h/cs)
cell_y    = (int(w/cs))*cell_x
total     = int(w/cs)*cell_x 

# Grid Map defined as an array of regions 
regions = gridmap_generator(h,w,cs)
# Arroy to store values of heatmap cells
map_values = np.zeros((int(w),int(h)))
initial_time = time.time()
to_time = time.time()
set_fps = 25            #Set maximum fps of the source video
prev_frame_time = 0     # variables to calculate fps
new_frame_time = 0


while True:
    while_running = time.time()
    new_time = while_running - initial_time
    if new_time >= 1/set_fps:
        ok, frame = cap.read()
        if ok:
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) 
            t  = new_frame_time-prev_frame_time
            prev_frame_time = new_frame_time
            fps = int(fps)
            
            # Resize source video
            frame = cv.resize(frame, (960, 540))

            # Human Detection
            (class_ids, scores, bboxes) = model.detect(frame,nmsThreshold=0.4,confThreshold = 0.5)
            # Human Tracking
            tracks = tracker.update(bboxes, scores, class_ids)
            
            for class_id, score, bbox,trk in zip(class_ids,scores,bboxes,tracks):
                x,y,w,h = bbox  
                if classes[class_id] == "person":
                    
                    cv.rectangle(frame,(x,y),(x+w,y+h),(136,249,248),3)
                    trk_id = trk[1]
                    xmin   = trk[2]
                    ymin   = trk[3]
                    width  = trk[4]
                    height = trk[5]
                    xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)

                    for i in range(len(regions)):
                        row = i//cell_x
                        col = i%cell_x
                        inside_region = cv.pointPolygonTest(regions[i],(int(xcentroid),int(ycentroid)),False)
                        if inside_region >0:
                            map_values[row*cs:row*cs+cs,col*cs:col*cs+cs] = np.clip(map_values[row*cs:row*cs+cs,col*cs:col*cs+cs]+(7*p2l(0.7)),0,255)

                
                    text = "ID {}".format(trk_id)
                    cv.putText(frame, text, (xcentroid - 10, ycentroid - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.circle(frame, (xcentroid, ycentroid), 4, (0, 255, 0), -1)


            for i in range(len(regions)):
                cv.polylines(frame,[regions[i]],isClosed=True,color=(255,255,255),thickness=1)
            
            heatmap = cv.applyColorMap(map_values.astype('uint8'),cv.COLORMAP_JET)

            # Show FPS and NN model name
            cv.rectangle(frame,(0,0),(150,30),(103,77,52),-1)
            cv.putText(frame, "FPS: {:d}".format(fps), (5, 20),cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)
            cv.rectangle(heatmap,(100,0),(260,30),(175,204,110),-1)
            cv.putText(frame, "Tiny Yolo v4".format(fps), (110, 20),cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)

            dst = cv.addWeighted(frame,0.7,heatmap,0.2,0)
            result.write(dst)
            cv.imshow('People Heatmap', dst)
            initial_time = while_running
        else:
            print('Video finished')
            break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 
        
cap.release()
result.release()
cv.destroyAllWindows()