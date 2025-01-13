import cv2 

import time 
import sys 
if len(sys.argv) < 2: 
    channel = 2
else: 
    channel = int(sys.argv[1])


cam_file = f"/dev/video{channel}"

DURATION = 5
NUM_SAVE = 5
from PIL import Image

cam = cv2.VideoCapture(cam_file)
print(cam)
import numpy as np 

for _ in range(100): 
    cam.read()

ret, frame = cam.read() 
print(ret)
cam.release() 
print(ret)
save_img = Image.fromarray(frame[..., ::-1])
save_img.save(f'./test.jpeg')
exit(0)

# cam.set()
    

# imgs = [] 
# start_time = time.time()
# prev_time = start_time
# while True: 
#     curr_time = time.time() 
#     if curr_time - prev_time < 1.0/30.0: 
#         continue 
#     # if curr_time - prev_time < 1.0: 
#     #     continue 
#     prev_time = curr_time
#     if curr_time - start_time > DURATION: 
#         break 
#     ret, frame = cam.read() 
#     if not ret: 
#         print(len(imgs), "failed")
#         break 
#     imgs.append(frame) 

# save_freq = len(imgs) // NUM_SAVE
# print(len(imgs))
# for i, frame  in enumerate(imgs): 
#     if i % save_freq: 
#         continue 
#     save_img = Image.fromarray(np.uint8(frame))
#     save_img.save(f'./{i}.jpeg')
    


# cHEIGHT))
# print(cam.get(cv2.CAP_PROP_FPS))
# ret, frame = cam.read() 
# if ret: 
#     cv2.imwrite("test_img.jpg", frame)
