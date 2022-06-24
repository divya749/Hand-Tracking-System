import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise

video = cv2.VideoCapture(0)
i = 0
_,frame = video.read()
prev = None
roi = cv2.selectROI(frame)
(x,y,w,h) = tuple(map(int, roi))
while True:
    _,frame = video.read()
    if i<60:
        i+=1
        if prev is None:
            prev = frame[y:y+h,x:x+w].copy()
            prev = np.float32(prev)
        else:
            cv2.accumulateWeighted(frame[y:y+h,x:x+w].copy(),prev,0.2)
    else:
        prev = cv2.convertScaleAbs(prev)
        prev_gray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)

        frame_gray = cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
        
        image = cv2.absdiff(prev_gray,frame_gray)

        _,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cont,hie = cv2.findContours(image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_new = image.copy()
        
        cont = max(cont,key=cv2.contourArea)
        conv_hull = cv2.convexHull(cont)
        cv2.drawContours(image,[conv_hull],-1,225,3)
        
        topmost = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
        bottommost = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
        leftmost = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
        rightmost = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
        centre_x = (leftmost[0]+rightmost[0])//2
        centre_y = (topmost[1]+bottommost[1])//2

        distance = pairwise.euclidean_distances([leftmost,rightmost,bottommost,topmost],[[centre_x,centre_y]])[0]
        radius = int(0.80*distance)
        
        circular_roi = np.zeros_like(image,dtype='uint8')
        cv2.circle(circular_roi,(centre_x,centre_y),radius,255,8)
        weighted_output = cv2.addWeighted(image.copy(),0.6,circular_roi,0.4,2)

        mask=cv2.bitwise_and(image_new,image,mask=circular_roi)
     
        cv2.rectangle(frame,(x,y),(x+w,y+h),255,3)
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
        cv2.imshow('weight',weighted_output)
        
    k=cv2.waitKey(5)
    if k==27:
        break
    
cap.release()
cv2.destroyAllWindows()