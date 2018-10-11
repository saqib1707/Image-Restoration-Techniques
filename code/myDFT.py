import numpy as np 
import cv2 
import math
import pdb
import time

img = cv2.imread('../data/fft_test.jpg', 0)
rows, cols = img.shape

img_dft = np.zeros((rows, cols), dtype=np.complex64)

start_time = time.time()
for i in range(0, rows):
	for j in range(0, cols):
		mysum = complex(0)
		for p in range(0, rows):
			for q in range(0, cols):
				temp = -2*np.pi*((i*p)/rows + (j*q)/cols)
				mysum += img[p,q]*(np.cos(temp) + np.sin(temp)*(1j))
		# pdb.set_trace()
		img_dft[i,j] = mysum

end_time = time.time()
img_dft_mag = abs(img_dft)
img_dft_mag = img_dft_mag.astype(np.uint8)
np.save('../data/fft_test_dft_mag.jpg', img_dft_mag)
time_req = (end_time - start_time)
print("Time taken for computing DFT = ", time_req)
print("Done")