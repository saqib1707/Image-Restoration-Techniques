"""
Author : Saqib Azim
Entire code was written by the author
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

img = cv2.imread('../data/monkey_small.png', 0)
img = img[:50,:50]
rows, cols = img.shape

# plt.imshow(img, cmap='gray')
# plt.show()

img_dft_slow = np.zeros((rows, cols), dtype=np.complex64)
img_dft_fast = np.zeros((rows, cols), dtype=np.complex64)
img_idft_slow = np.zeros((rows, cols), dtype=np.complex64)
img_idft_fast = np.zeros((rows, cols), dtype=np.complex64)

start_time = time.time()

# slow DFT crude approach
# ------------------------------------------------
for i in range(0, rows):
	for j in range(0, cols):
		# print(i,j)
		mysum = complex(0)
		for p in range(0, rows):
			for q in range(0, cols):
				mysum += img[p,q]*np.exp(-2j*np.pi*((i*p)/rows + (j*q)/cols))
		img_dft_slow[i,j] = mysum
# ------------------------------------------------

# vectorized DFT approach
# ------------------------------------------------
mat1 = np.zeros((cols, cols), dtype=np.complex64)
mat2 = np.zeros((rows, rows), dtype=np.complex64)
for i in range(cols):
	for j in range(cols):
		mat1[i,j] = i*j

mat1 = np.exp((-2j*np.pi*mat1)/cols)
img_dft_fast = np.matmul(img, mat1)

for i in range(rows):
	for j in range(rows):
		mat2[i,j] = i*j

mat2 = np.exp((-2j*np.pi*mat2)/rows)
img_dft_fast = np.matmul(mat2, img_dft_fast)
# ------------------------------------------------

# slow IDFT crude approach
# ------------------------------------------------
for i in range(0, rows):
	for j in range(0, cols):
		mysum = complex(0)
		for p in range(0, rows):
			for q in range(0, cols):
				mysum += img_dft_slow[p,q]*np.exp(2j*np.pi*((i*p)/rows + (j*q)/cols))
		img_idft_slow[i,j] = mysum/(rows*cols)
# ------------------------------------------------

# vectorized IDFT approach
# ------------------------------------------------
mat1 = np.zeros((cols, cols), dtype=np.complex64)
mat2 = np.zeros((rows, rows), dtype=np.complex64)
for i in range(cols):
	for j in range(cols):
		mat1[i,j] = i*j

mat1 = np.exp((2j*np.pi*mat1)/cols)
img_idft_fast = np.matmul(img_dft_fast, mat1)

for i in range(rows):
	for j in range(rows):
		mat2[i,j] = i*j

mat2 = np.exp((2j*np.pi*mat2)/rows)
img_idft_fast = np.matmul(mat2, img_idft_fast)/(rows*cols)
# ------------------------------------------------

end_time = time.time()

img_dft_slow_mag = np.log10(abs(img_dft_slow))
img_dft_fast_mag = np.log10(abs(img_dft_fast))
img_idft_slow_mag = np.real(img_idft_slow)
img_idft_fast_mag = np.real(img_idft_fast)


fig = plt.figure(1)
plt.subplot(221);plt.grid(False);plt.axis("off");plt.title("DFT Slow Method");plt.imshow(img_dft_slow_mag, cmap='gray')
plt.subplot(222);plt.grid(False);plt.axis("off");plt.title("DFT Vectorized Method");plt.imshow(img_dft_fast_mag, cmap='gray')
plt.subplot(223);plt.grid(False);plt.axis("off");plt.title("IDFT Slow Method");plt.imshow(img_idft_slow_mag, cmap='gray')
plt.subplot(224);plt.grid(False);plt.axis("off");plt.title("IDFT Vectorized Method");plt.imshow(img_idft_fast_mag, cmap='gray')
plt.show()


# cv2.imwrite('../plots-results/experiment-3/bird_dft_slow.png', img_dft_slow_mag)
# cv2.imwrite('../plots-results/experiment-3/bird_dft_fast.png', img_dft_fast_mag)

time_req = (end_time - start_time)
print("Time taken for computing DFT = ", time_req)
print("Done")