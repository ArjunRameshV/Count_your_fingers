import cv2
import numpy as np
from sklearn.metrics import pairwise

##global variables
background = None
accumulated_weight = 0.5
top = 20
bottom = 300
right = 300
left = 600

##to fing avg background value
def accumulated_avg(frame,accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    #weighted sum of the input src
    cv2.accumulateWeighted(frame,background,accumulated_weight)
    #return is not necessary, changes happen to the background variable

##segmenting the hand in the roi
def segment(frame,threshold=25):
    diff = cv2.absdiff(background.astype("uint8"),frame)
    ret, thresh = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)
    (cnts,_) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None
    else :
        #assuming the largest external contour is the hand
        hand_segment = max(cnts,key=cv2.contourArea)
        return (thresh,hand_segment)

##few assumptions, that the hand comes straightup
# find extreme points in the hand convex hull polygon, the top bottom left and right
# their intersection - center of the hand
# distance from a point furthest from the center
# draw a circle from the center with radius about 90% of that distance
# points outside the circle will be extended fingers
###########################

def count_fingers(thresh,hand_segment):
    #calculate the convex hull
    conv_hull = cv2.convexHull(hand_segment)
    #top point
    roi_top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    #bottom point
    roi_bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    #left point
    roi_left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    #right point
    roi_right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])

    cx = (roi_left[0] + roi_right[0]) // 2
    cy = (roi_top[1] + roi_bottom[1]) // 2
    # print(cx,cy,"the centers")
    # print(roi_top,'top')
    # print(roi_left,'left')
    # print(roi_right,'right')
    # print(roi_bottom,'bottom')
    #euclidean distance
    distance = pairwise.euclidean_distances([[cx,cy]],Y=[roi_left,roi_right,roi_top,roi_bottom])[0]
    max_distance = distance.max()

    radius = int(0.75*max_distance)
    circumference = (2*np.pi*radius)

    circular_roi = np.zeros_like(thresh,dtype='uint8')
    # print(circular_roi.shape,"the original dim")
    cv2.circle(circular_roi,(cx,cy),radius,255,10)
    # print(circular_roi.shape,thresh.shape)
    circular_roi = cv2.bitwise_and(thresh,thresh,mask=circular_roi)
    cnts,_ = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    count = 0
    for cnt in cnts:
        (x,y,w,h) = cv2.boundingRect(cnt)
        #make sure to not consider things too far below the center
        wrist = (cy+(cy*0.10)) > (y+h)
        #number of points along the contour doesnt add to be more than 25% of the circumference,
        #  as they be mostly be some external contour
        limit_points = ((circumference*0.10) > cnt.shape[0])

        if wrist and limit_points:
            count += 1

    return count

cap = cv2.VideoCapture(0)

num_frames = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame_copy = frame.copy()
    roi = frame[top:bottom,right:left]

    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)

    if num_frames < 60:
        #to get the background initially
        accumulated_avg(gray,accumulated_weight)

        if num_frames <= 59:
            cv2.putText(frame_copy,"Wait, getting background",(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow("finger count",frame_copy)
    else:
        hand = segment(gray)
        # we might receive none from segment, so we need to check first and then unpack the tuple
        if hand is not None:
            thresh, hand_segment = hand
            #live time contour around the hand
            cv2.drawContours(frame_copy,[hand_segment+(right,top)],-1,(255,0,0),5)

            fingers = count_fingers(thresh,hand_segment)
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('thresholded',thresh)

    cv2.rectangle(frame_copy,(left,top),(right,bottom),(0,0,255),5)
    num_frames += 1

    cv2.imshow("finger count",frame_copy)

    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
