from yoloPydarknet import pydarknetYOLO
import cv2
import imutils
import time
import numpy as np

drawThreshold = 12  #counts for really painting

yolo = pydarknetYOLO(obdata="cfg.paintOnAir/obj.data", \
    weights="cfg.paintOnAir/yolov3_40000.weights", \
    cfg="cfg.paintOnAir/yolov3.cfg")
video_out = "realtime.avi"

start_time = time.time()

#store the size of the webcam.
width = 0
height = 0
lastGesture = '0'
lastDrawPoint = (0,0)
modeNow = ""
colorNow = 0
accDrawCount = 0
paintColor = [(0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255)]

def newLayer():
    # Create a blank 300x300 black image
    return np.zeros((height, width, 3), np.uint8)

def drawPoint(img, color, bold, point):
    cv2.circle(img, point, bold, color, -1)
    return img

def drawLine(img, color, bold, toPoint):
    cv2.line(img, lastDrawPoint, toPoint, color, bold)
    return img

if __name__ == "__main__":

    VIDEO_IN = cv2.VideoCapture(0)
    VIDEO_IN.set(3,1024)
    VIDEO_IN.set(4,768)

    if(video_out!=""):
        width = int(VIDEO_IN.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(VIDEO_IN.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_out,fourcc, 15.0, (int(width),int(height)))


    layerPaint = newLayer()
    frameID = 0
    while True:
        hasFrame, frame = VIDEO_IN.read()
        frame = cv2.flip(frame, 1)
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("--- %s seconds ---" % (time.time() - start_time))
            break

        labels, scores, boxes = yolo.getObject(frame, labelWant="", drawBox=True, bold=1, textsize=0.85, bcolor=(255,255,255), tcolor=(0,255,0))

        if(len(labels)>0):
            if(labels[0]=='fingertip'):
                accDrawCount += 1
                if(accDrawCount>drawThreshold):
                    modeNow = "painting"
                    center_x = boxes[0][0] + int(boxes[0][2]/2)
                    if(lastDrawPoint[0] == 0 and lastDrawPoint[1] == 0):
                        layerPaint = drawPoint(layerPaint, paintColor[colorNow], 5, (center_x, boxes[0][1]))
                    else:
                        layerPaint = drawLine(layerPaint, paintColor[colorNow], 5, (center_x, boxes[0][1]))

                    lastDrawPoint = (center_x, boxes[0][1])
                    print("draw it.")

            else:
                accDrawCount = 0
                lastDrawPoint = (0,0)
                if(labels[0]=='palm'):
                    modeNow = "eraser"
                    center_x = boxes[0][0] + int(boxes[0][2]/2)
                    center_y = boxes[0][1] + int(boxes[0][3]/2)
                    layerPaint = drawPoint(layerPaint, (0,0,0), int(boxes[0][2]/2), (center_x, center_y))
                    print("clear it.")
                #elif(labels[0]=='3'):
                #    modeNow = "change color"
                #    if(lastGesture != labels[0]):
                #        colorNow += 1
                #        if(colorNow>len(paintColor)-1):
                #            colorNow = 0

                #        print("color: {}".format(paintColor[colorNow]))

            lastGesture = labels[0]

        cv2.putText(frame, "Mode:", (int(width/2 - 120), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        cv2.putText(frame, modeNow, (int(width/2 - 20), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, paintColor[colorNow], 2)

        overlapping = cv2.addWeighted(layerPaint,0.5,frame, 0.5,0)
        #overlapping = cv2.add(frame, layerPaint) 
        cv2.imshow("Frame", imutils.resize(overlapping, width=950))
        #cv2.imshow("Draw", imutils.resize(layerPaint, width=950))
        #print ("Object counts:", yolo.objCounts)
        if(video_out!=""):
            out.write(overlapping)


        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            out.release()
            break

