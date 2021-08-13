import cv2
import numpy as np
import random
import torch
import math
from imutils import perspective
from PIL import Image
import pandas as pd
# from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

coco_names = [
    '__background__','s1','s2'
]

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices) #score是按照大小排序
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    labels = labels[:thresholded_preds_count]
    return masks, boxes, labels

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    all_area=[]
    all_center=[]
#     COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
    df = pd.DataFrame(columns=['class', 'angle', 'area'])
    for i in range(len(masks)):
#         print(masks[i].shape)
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = random.sample(range(0, 255), 3)
        ##面積
#         area = len(np.where(masks[i]==1)[0])
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
#         print(red_map[i].shape)
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        ##最小矩形框
        img = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)
        
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        area, angle, box, center = draw_min_rect( contours, color)
#         print(i)
#         print("boxes:", np.array(boxes))
        
        left = get_head(boxes[i], masks[i])
#         cv2.circle(image, boxes[i][0], radius=5, color=(0,0,255), thickness=-1)
#         cv2.circle(image, boxes[i][1], radius=5, color=(0,255,0), thickness=-1)
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.drawContours(image, [box], 0, color, 3)
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                      thickness=2)
#         text = labels[i]+" area:"+str(area)
#         points = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)) #藍,綠,紅,黃
       
        for j in left:
            cv2.circle(image, (j[0],j[1]), radius=5, color=(0,0,255), thickness=-1)
       
        # put the label text above the objects
        cv2.putText(image , str(i), (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
#         detail = "No.%d, class: "
        df.loc[i]=[labels[i], str(angle), str(area)]
        all_area.append(area)
        all_center.append(center)

    print(df)
    return image

def draw_min_rect(cnts, color):  # conts = contours

    if(len(cnts)!=1):
        max = -1
        for c in cnts:
            if(len(c)>max):
                max=len(c)
                cnt=c
    else:
        cnt = cnts[0]
    area = cv2.contourArea(cnt)
    min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle

    angle = min_rect[2]
    angle = round(angle,2)

    min_rect = cv2.boxPoints(min_rect)
#     print("min:", min_rect)
    box = perspective.order_points(min_rect)
    box = np.int0(box)
    (tl, tr, br, bl) = box
    print("min:",tl, type(tl))
    
    cx = (tr[0] + bl[0])//2
    cy = (tr[1] + bl[1])//2

    return area, angle, box, (cx, cy)
def get_head2(box, mask):
    tl = np.asarray(box[0])
    br = np.asarray(box[1])
    tr = np.asarray((box[1][0], box[0][1]))
    bl = np.asarray((box[0][0], box[1][1]))
    print(tl, type(tl))
    print(bl, type(bl))
    line1 = length(tl, tr)
    line2 = length(tl, bl)
    
def get_head(box, mask):
     #以中線切割兩部分，比較多像素點的那一側為頭
#     box = np.array(box)
    
#     box = perspective.order_points(box)
    
    tl = np.asarray(box[0])
    br = np.asarray(box[1])
    tr = np.asarray((box[1][0], box[0][1]))
    bl = np.asarray((box[0][0], box[1][1]))
    print(tl)
    print(bl)
    line1 = length(tl, tr)
    line2 = length(tl, bl)
    if(line1>line2):
        x1, y1 = forthpoint(tl, tr) ##1/2
        x2, y2 = forthpoint(bl, br)
        x3, y3 = forthpoint(tr, tl)
        x4, y4 = forthpoint(br, bl)
        
#         x3, y3 = midpoint()##1/4r
    else:
        x1, y1 = forthpoint(tl, bl)
        x2, y2 = forthpoint(tr, br)
        x3, y3 = forthpoint(bl, tl)
        x4, y4 = forthpoint(br, tr)
#     return (int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))
    #求線
    mask = np.where(mask==1)
    left1=[]
    right1=[]
    left2=[]
    right2=[]
    l1=0
    r1=0
    l2=0
    r2=0
    print("p1:", x1, y1)
    print("p2:", x2, y2)
    for i in range(len(mask[0])): 
   
        x = mask[1][i]
        y = mask[0][i]
        if(y1>y2):
            tmp1 = (y1-y2)*x + (x2-x1)*y + x1*y2 - x2*y1
        else:
            tmp1 = (y2-y1)*x + (x1-x2)*y + x2*y1 - x1*y2
            
        if(tmp1<0):
            left1.append([x,y])
            l1 += l1
        elif(tmp1>0):
            right1.append([x,y])
            r1 +=r1
            
        if(y3>y4):
            tmp2 = (y3-y4)*x + (x4-x3)*y + x3*y4 - x4*y3
        else:
            tmp2 = (y4-y3)*x + (x3-x4)*y + x4*y3 - x3*y4
        if(tmp2<0):
            left2.append([x,y])
            l2 += l2
        elif(tmp2>0):
            right2.append([x,y])
            r2 +=r2
    print("l1:", len(left1), "r1:", len(right1))
    if(len(left1)<len(right1)):
        p1 = left1
    else:
        p1 = right1
    if(len(left2)<len(right2)):
        p2 = left2
    else:
        p2 = right2
    if(p1>p2):
        return p1
    else:
        return p2
#         print("right")
#     return p1,p2
        
#         break
#     print("all point:", len(mask[0]))
#     print("left:", left)
#     print("right:", right)
   

#     return left, right
def get_angle(box,angle):
    
    if(box[0][1]==box[1][1]): ##0度或180度
        line1 = box[0]-box[1]
        line2 = box[0]-box[3]
        len1 = math.hypot(line1[0],line1[1])
        len2 = math.hypot(line2[0],line2[1])
        if(len1>len2):
            return 0
        else:
            return 180
    line1 = box[0]-box[1]
    line2 = box[0]-box[3]
    len1 = math.hypot(line1[0],line1[1])
    len2 = math.hypot(line2[0],line2[1])
    angle = 0-angle
    if(len1>len2):
        angle = angle-90
#     angle = 0-angle
#     print(angle)
#     if(angle==-180.0):
#         angle=0
#     if(angle>90):
#         angle = angle - 180
    angle = round(angle,2)
    return angle

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def forthpoint(ptA, ptB):
    return ((ptA[0]*0.25 + ptB[0]*0.75) , (ptA[1]*0.25 + ptB[1]*0.75))

def length(ptA, ptB):
    line = ptA-ptB
    return math.hypot(line[0],line[1])
    

        
        