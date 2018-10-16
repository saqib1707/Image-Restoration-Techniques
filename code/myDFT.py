import numpy as np
import cv2
import time

img = cv2.imread('../data/Blurry1_1.jpg', 0)
img = img[:50,:50]
rows, cols = img.shape

img_dft1 = np.zeros((rows, cols), dtype=np.complex64)
img_dft2 = np.zeros((rows, cols), dtype=np.complex64)

start_time = time.time()

# first crude approach
# for i in range(0, rows):
# 	for j in range(0, cols):
# 		mysum = complex(0)
# 		for p in range(0, rows):
# 			for q in range(0, cols):
# 				mysum += img[p,q]*np.exp(-2j*np.pi*((i*p)/rows + (j*q)/cols))
# 		img_dft2[i,j] = mysum

# vectorized dft approach
mat1 = np.zeros((cols, cols), dtype=np.complex64)
mat2 = np.zeros((rows, rows), dtype=np.complex64)
for i in range(cols):
	for j in range(cols):
		mat1[i,j] = i*j

mat1 = np.exp((-2j*np.pi*mat1)/cols)
img_dft1 = np.matmul(img, mat1)

for i in range(rows):
	for j in range(rows):
		mat2[i,j] = i*j

mat2 = np.exp((-2j*np.pi*mat2)/rows)
img_dft1 = np.matmul(mat2, img_dft1)

end_time = time.time()
img_dft1_mag = abs(img_dft1)
img_dft1_mag = img_dft1_mag.astype(np.uint8)
# img_dft2_mag = abs(img_dft2)
# img_dft2_mag = img_dft2_mag.astype(np.uint8)

# cv2.imshow('bada dft', img_dft2_mag)
# cv2.imshow('chota dft', img_dft1_mag)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# np.save('../data/fft_test_dft_mag.jpg', img_dft1_mag)
time_req = (end_time - start_time)
print("Time taken for computing DFT = ", time_req)
print("Done")