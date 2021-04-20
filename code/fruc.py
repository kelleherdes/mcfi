import cv2
from new_motion_interp import predict_frame_uni
import numpy as np

input_video = '../videos/stefan.mp4'
cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
ret, frame = cap.read()
out = cv2.VideoWriter('ballet.avi', fourcc, 30, (frame.shape[1], frame.shape[0]))
started = False
i = 0
out.write(frame)

while(ret):
    print("Frame: ", i)
    i += 1
    frame1 = frame
    ret, frame = cap.read()
    frame2 = frame
    if ret==True:
        interp = predict_frame_uni(frame1, frame2, 3, 5)
        out.write(interp.astype(np.uint8))

    out.write(frame2)

cap.release()
out.release()




