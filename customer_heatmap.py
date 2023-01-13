#!/usr/bin/python3
import sys
import numpy as np
import cv2 as cv
import time
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from utils import gridmap_generator


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

#Source video
cap = cv.VideoCapture(str(sys.argv[1]))
cap.set(cv.CAP_PROP_FRAME_WIDTH,960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,540)
h = 960
w = 540

#Resulting video
result = cv.VideoWriter(str(sys.argv[2]+".mp4"),cv.VideoWriter_fourcc(*'MJPG'),25, (960,540))

#Grid Map 
cs = 60   # Cell_size
cell_x    = int(h/cs)
cell_y    = (int(w/cs))*cell_x
total     = int(w/cs)*cell_x 

# Grid Map defined as an array of regions 
regions = gridmap_generator(h,w,cs)

map_counter = np.zeros((int(w),int(h)))
# Array to store spend time per cell
map_time    = np.zeros((int(w/cs),cell_x))
# Array to store IDS
map_id      = np.array([set() for _ in range(total)]).reshape((int(w/cs),cell_x)) 
map_f      = np.array([set() for _ in range(total)]).reshape((int(w/cs),cell_x)) 

initial_time = time.time()
to_time = time.time()
set_fps = 25         #Set maximum fps of the source video
prev_frame_time = 0  # variables to calculate fps
new_frame_time = 0

min_time = 3          # Minimum time to include a person in the counter

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

            # Draw Gridmap divisions over output
            for i in range(len(regions)):
                cv.polylines(frame,[regions[i]],isClosed=True,color=(255,255,255),thickness=1)
            
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
                        if map_time[row,col]>100:
                            map_time[row,col] = 0
                            
                        if inside_region>0 and trk_id not in map_id[row,col]:
                            map_id[row,col].add(trk_id)
                            map_time[row,col] = 0
                            
                        if trk_id in map_id[row,col] and inside_region>0:
                            map_time[row,col] += t
                            
                            if map_time[row,col]>min_time and map_time[row,col]<100:
                                map_f[row,col].add(trk_id)
                                map_counter[row*cs:row*cs+cs,col*cs:col*cs+cs] = len(map_f[row,col])
                                map_time[row,col] = 0
                                
                
                    text = "ID {}".format(trk_id)
                    cv.putText(frame, text, (xcentroid - 10, ycentroid - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.circle(frame, (xcentroid, ycentroid), 4, (0, 255, 0), -1)
            
            map_heat = (map_counter/(map_counter.max()+0.01))*255
            heatmap = cv.applyColorMap(map_heat.astype('uint8'),cv.COLORMAP_JET)
            overheat = cv.applyColorMap(map_heat.astype('uint8'),cv.COLORMAP_JET)
            
            for i in range(len(regions)):
                row = i//cell_x
                col = i%cell_x
                cv.polylines(heatmap,[regions[i]],isClosed=True,color=(255,255,255),thickness=1)
                cv.putText(heatmap, "{:d}".format(len(map_f[row,col])), (col*cs+25, row*cs+25),cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255,255), 1)
                cv.putText(frame, "{:.1f}".format(map_time[row,col]), (col*cs+10, row*cs+20),cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255,255), 1)

            # Show FPS and NN model name
            cv.rectangle(frame,(0,0),(150,30),(103,77,52),-1)
            cv.putText(frame, "FPS: {:d}".format(fps), (5, 20),cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)
            cv.rectangle(frame,(100,0),(260,30),(175,204,110),-1)
            cv.putText(frame, "Tiny Yolo v4".format(fps), (110, 20),cv.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 255), 1)

            dst = cv.addWeighted(frame,0.9,overheat,0.4,0)
            result.write(dst)
            cv.imshow('Customer Heatmap', heatmap)
            cv.imshow("Human detection and tracking",dst)
            initial_time = while_running
        else:
            print('Video finished')
            break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 
        
cap.release()
result.release()
cv.destroyAllWindows()