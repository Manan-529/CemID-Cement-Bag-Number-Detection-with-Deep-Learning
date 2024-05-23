import cv2
import numpy as np
import time
import pandas as pd
import tensorflow as tf

# Define constants
COUNT = 192 * 640
BAGTHRESH = 0.08
NUMTHRESH = 0.01

# Preprocess function
def preProcess(frame):
    frame = cv2.resize(frame, (640, 192))
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(frame2, 0) / 255
    return inp

# Bag count function
def bagCount(background, count=COUNT, backThreshold=0.95):
    indsBg = np.where(background > backThreshold)
    countBag = count - len(indsBg[0])
    return countBag, countBag / count, indsBg

# Number count function
def numCount(nums, count, numThreshold=0.6):
    indsNums = np.where(nums > numThreshold)
    return len(indsNums[0]), len(indsNums[0]) / count, indsNums

# Overlay function
def overLay(frame, bag, num, th1=1, th2=0.7, th3=0.8):
    frame[:, :, 0] = cv2.addWeighted(frame[:, :, 0], th1, bag, th2, 0)
    frame[:, :, 2] = cv2.addWeighted(frame[:, :, 2], th1, num, th3, 0)
    return frame

# Load model
segnetModel = tf.keras.models.load_model('../model/segnet.h5')

# Define path to video
path = '../data/vdos/IMG_0422.MOV'

print(path)
maxBagIou = 0
maxNumIou = 0
numIou = 0
flag_arr = []
flag = False
i = 0
check = True
time_arr = []

try:
    vdo = cv2.VideoCapture(path)
    while vdo.isOpened():
        start = time.time()
        ret, frame = vdo.read()
        if i % 10 == 0:
            i = 0
            if ret:
                inp = preProcess(frame)
                pred = segnetModel.predict(inp)
                bagCnt, bagIou, _ = bagCount(pred[0, :, :, 0])
                flag = False
                if bagIou >= BAGTHRESH:
                    if bagIou >= maxBagIou:
                        maxBagIou = bagIou
                    elif bagIou < maxBagIou and check:
                        numCnt, numIou, _ = numCount(pred[0, :, :, 2], bagCnt)
                        if numIou >= NUMTHRESH:
                            flag_arr.append(1)
                            flag = True
                        else:
                            flag_arr.append(0)
                            flag = False
                        check = False
                else:
                    maxBagIou = 0
                    numIou = 0
                    check = True

                annotated = cv2.putText(inp[0], f'{str(bagIou)[:4]} {str(numIou)[:4]} {flag}', (10, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                overlayed = overLay(inp[0].astype('float32'), pred[0, :, :, 1].astype('float32'),
                                    pred[0, :, :, 2].astype('float32'))
                cv2.imshow('frame', overlayed)
            else:
                break
        end = time.time()
        time_arr.append(end - start)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()

print(f"Mean Time Per Prediction Per Frame {np.mean(time_arr)} seconds")

# Save flags to CSV
flag_path = '../data/IMG_0421.csv'
df = pd.DataFrame({'Flags': flag_arr})
df.to_csv(flag_path, index=False)

print(flag_arr)
