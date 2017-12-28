'''
Method2

Using Speeded-Up Robust Features(SURF) or Scale-Invariant Feature Transform(SIFT)
and Fast Library for Brute Force Matcher(BFMatcher)
'''
import numpy as np
import cv2
MIN_MATCH_COUNT=15

ver = (cv2.__version__).split('.')

#Change to SIFT to SURF to use Speeded-Up Robust Features
if int(ver[0]) < 3:
	detector=cv2.SIFT()
else:
	detector=cv2.xfeatures2d.SIFT_create()

bf = cv2.BFMatcher()

#The path of the image
trainImg=cv2.imread("path/to/pic.png",0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    
    matches = bf.knnMatch(queryDesc,trainDesc,k=2)

    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
    else:
        print ("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()