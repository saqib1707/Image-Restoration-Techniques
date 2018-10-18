"""
Author : Saqib Azim
Entire code was written by the author
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.measure import compare_ssim as ssim
import cv2
import pdb
import argparse
from scipy import ndimage

psnr_list = []
ssim_list = []
parameter_list = []

def mypsnr(image1, image2):
    R = 1.0
    image1 = cv2.cvtColor(cv2.normalize(image1, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F), cv2.COLOR_BGR2YCR_CB)
    image2 = cv2.cvtColor(cv2.normalize(image2, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F), cv2.COLOR_BGR2YCR_CB)
    image1 = image1[:,:,0]
    image2 = image2[:,:,0]
    mse = np.mean(np.square(image1 - image2))
    psnr = 10*np.log10(np.square(R)/mse)
    return psnr

def myssim(image1, image2, k=(0.01, 0.03), l=255):
    c1 = np.power(k[0]*l, 2)
    c2 = np.power(k[1]*l, 2)
    mu_image1 = np.mean(image1[:,:,0])
    mu_image2 = np.mean(image2[:,:,0])
    var_image1 = np.var(image1[:,:,0])
    var_image2 = np.var(image2[:,:,0])
    cov_image1_image2 = np.mean((image1[:,:,0] - mu_image1)*(image2[:,:,0] - mu_image2))
    return ((2*mu_image1*mu_image2 + c1)*(2*cov_image1_image2 + c2))/((mu_image1**2 + mu_image2**2 + c1)*(var_image1 + var_image2 + c2))

def inbuilt_ssim(image1, image2):
    # image1 = cv2.normalize(image1, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    # image2 = cv2.normalize(image2, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
    return ssim(image1, image2, multichannel=True)

def mybutterworth(rows, cols, D0=70.0):
    order = 5
    D = np.zeros((rows, cols), dtype=np.float32)
    butterworth_filter = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            D[i,j] = np.sqrt(np.square(i - rows/2) + np.square(j - cols/2))
    butterworth_filter = 1/(1 + np.power(D/D0, 2*order))
    return butterworth_filter

class imageRestoration():
    def __init__(self, question):
        self.original_image = cv2.imread('../data/original_image.jpg')
        self.blurred_image = cv2.imread('../data/Blurry1_1.jpg')
        self.kernel = cv2.imread('../data/small_kernel_1.jpg', 0)

        self.original_image = cv2.normalize(self.original_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        self.blurred_image = cv2.normalize(self.blurred_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        self.kernel = cv2.normalize(self.kernel, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)

        self.image_size = self.blurred_image.shape
        self.kernel_size = self.kernel.shape
        if(question == 1):
            self.full_inverse_filter()
        elif question == 2:
            self.truncated_inverse_filter()
        elif question == 3:
            self.wiener_filter()
        elif question == 4:
            self.constrained_ls_filter()

    def get_padded_size(self):
        self.new_row = self.image_size[0] + self.kernel_size[0] - 1
        self.new_col = self.image_size[1] + self.kernel_size[1] - 1

        if self.new_row%2 != 0:
            self.new_row += 1
        if self.new_col%2 != 0:
            self.new_col += 1
        return (self.new_row, self.new_col)

    def initial_common_section(self):
        self.new_row, self.new_col = self.get_padded_size()
        new_kernel = np.zeros((self.new_row, self.new_col), dtype=np.float32)
        new_blurred_image = np.zeros((self.new_row, self.new_col, 3), dtype=np.float32)
        new_kernel[0:self.kernel_size[0], 0:self.kernel_size[1]] = self.kernel
        new_blurred_image[0:self.image_size[0], 0:self.image_size[1], :] = self.blurred_image

        self.blurred_image_fft = np.zeros((self.new_row, self.new_col, 3), dtype=np.complex64)
        self.blurred_image_shifted_fft = np.zeros((self.new_row, self.new_col, 3), dtype=np.complex64)

        self.kernel_fft = np.fft.fft2(new_kernel)
        self.kernel_shifted_fft = np.fft.fftshift(self.kernel_fft)
        for i in range(3):
            self.blurred_image_fft[:,:,i] = np.fft.fft2(new_blurred_image[:,:,i])
            self.blurred_image_shifted_fft[:,:,i] = np.fft.fftshift(self.blurred_image_fft[:,:,i])

        self.estimated_image_fft = np.zeros((self.new_row, self.new_col, 3), dtype=np.complex64)
        self.estimated_image = np.zeros((self.new_row, self.new_col, 3), dtype=np.complex64)

    def full_inverse_filter(self):
        print("Full Inverse Filtering")
        self.initial_common_section()
        for i in range(3):
            self.estimated_image_fft[:,:,i] = np.divide(self.blurred_image_shifted_fft[:,:,i], self.kernel_shifted_fft)
            self.estimated_image[:,:,i] = np.fft.ifft2(np.fft.ifftshift(self.estimated_image_fft[:,:,i]))

        recovered_image = np.abs(self.estimated_image[0:self.image_size[0], 0:self.image_size[1], :])
        recovered_image = cv2.cvtColor(cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F), cv2.COLOR_BGR2RGB)

        fig = plt.figure()
        plt.axis("off")
        plt.imshow(recovered_image)
        plt.show()
        plt.imsave('../plots-results/experiment-3/full_inverse_image_1.png', recovered_image)
        print("PSNR = ", mypsnr(self.original_image, recovered_image))
        print("SSIM = ", inbuilt_ssim(self.original_image, recovered_image))

    def truncated_inverse_filter(self):
        print("Truncated Inverse Filtering")
        self.initial_common_section()

        D0_min = 1.0
        D0_max = 200.0
        # D0_init = np.mean(np.square(np.abs(self.kernel_shifted_fft)))
        D0_init = 105.0
        fig = plt.figure(figsize=(9,7))

        butterworth_filter = mybutterworth(self.new_row, self.new_col, D0=D0_init)
        for i in range(3):
            self.estimated_image_fft[:,:,i] = np.multiply(np.divide(self.blurred_image_shifted_fft[:,:,i], self.kernel_shifted_fft), butterworth_filter)
            self.estimated_image[:,:,i] = np.fft.ifft2(np.fft.ifftshift(self.estimated_image_fft[:,:,i]))

        recovered_image = np.abs(self.estimated_image[0:self.image_size[0], 0:self.image_size[1], :])
        recovered_image = cv2.cvtColor(cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F), cv2.COLOR_BGR2RGB)

        psnr_list.append(mypsnr(self.original_image, recovered_image))
        ssim_list.append(inbuilt_ssim(self.original_image, recovered_image))
        parameter_list.append(D0_init)

        plt.axis("off")
        recovered_image_plot = plt.imshow(recovered_image)

        plt.imsave('../plots-results/experiment-3/truncated_image_1.png', recovered_image)

        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
        D0_slider = Slider(slider_ax, 'D0', D0_min, D0_max, valinit=D0_init)

        def update(D0):
            butterworth_filter = mybutterworth(self.new_row, self.new_col, D0=D0)
            for i in range(3):
                self.estimated_image_fft[:,:,i] = np.multiply(np.divide(self.blurred_image_shifted_fft[:,:,i], self.kernel_shifted_fft), butterworth_filter)
                self.estimated_image[:,:,i] = np.fft.ifft2(np.fft.ifftshift(self.estimated_image_fft[:,:,i]))
            
            recovered_image = np.abs(self.estimated_image[0:self.image_size[0], 0:self.image_size[1], :])
            recovered_image = cv2.cvtColor(cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F), cv2.COLOR_BGR2RGB)
            psnr_list.append(mypsnr(self.original_image, recovered_image))
            ssim_list.append(inbuilt_ssim(self.original_image, recovered_image))
            parameter_list.append(D0)
            recovered_image_plot.set_data(recovered_image)
            fig.canvas.draw_idle()

        D0_slider.on_changed(update)
        plt.show()

    def wiener_filter(self):
        print("Wiener Filtering")
        self.initial_common_section()

        K_min = 1.0
        K_max = 500
        # K_init = np.mean(np.square(np.abs(self.kernel_shifted_fft)))
        K_init = 40.0
        fig = plt.figure(figsize=(9,7))

        temp = np.multiply(np.conjugate(self.kernel_shifted_fft), self.kernel_shifted_fft)
        for i in range(3):
            self.estimated_image_fft[:,:,i] = np.multiply(np.divide(temp, np.multiply(self.kernel_shifted_fft, temp + K_init)), self.blurred_image_shifted_fft[:,:,i])
            self.estimated_image[:,:,i] = np.fft.ifft2(np.fft.ifftshift(self.estimated_image_fft[:,:,i]))

        recovered_image = np.abs(self.estimated_image[0:self.image_size[0], 0:self.image_size[1], :])
        recovered_image = cv2.cvtColor(cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F), cv2.COLOR_BGR2RGB)

        psnr_list.append(mypsnr(self.original_image, recovered_image))
        ssim_list.append(inbuilt_ssim(self.original_image, recovered_image))
        parameter_list.append(K_init)

        plt.axis("off")
        recovered_image_plot = plt.imshow(recovered_image)
        plt.imsave('../plots-results/experiment-3/wiener_image_1.png', recovered_image)

        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
        K_slider = Slider(slider_ax, 'K', K_min, K_max, valinit=K_init)

        def update(K):
            temp = np.multiply(np.conjugate(self.kernel_shifted_fft), self.kernel_shifted_fft)
            for i in range(3):
                self.estimated_image_fft[:,:,i] = np.multiply(np.divide(temp, np.multiply(self.kernel_shifted_fft, temp + K)), self.blurred_image_shifted_fft[:,:,i])
                self.estimated_image[:,:,i] = np.fft.ifft2(np.fft.ifftshift(self.estimated_image_fft[:,:,i]))

            recovered_image = np.abs(self.estimated_image[0:self.image_size[0], 0:self.image_size[1], :])
            recovered_image = cv2.cvtColor(cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F), cv2.COLOR_BGR2RGB)
            psnr_list.append(mypsnr(self.original_image, recovered_image))
            ssim_list.append(inbuilt_ssim(self.original_image, recovered_image))
            parameter_list.append(K)
            recovered_image_plot.set_data(recovered_image)
            fig.canvas.draw_idle()

        K_slider.on_changed(update)
        plt.show()

    def constrained_ls_filter(self):
        print("Constrained Least Square Filtering")
        self.initial_common_section()

        new_p = np.zeros((self.new_row, self.new_col), dtype=np.float32)
        new_p[0:3, 0:3] = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        p_fft = np.fft.fft2(new_p)
        p_shifted_fft = np.fft.fftshift(p_fft)

        gamma_min = 1.0
        gamma_max = 500.0
        # gamma_init = np.mean(np.square(np.abs(self.kernel_shifted_fft)))
        gamma_init = 135.0
        fig = plt.figure(figsize=(9,7))

        temp1 = np.multiply(np.conjugate(self.kernel_shifted_fft), self.kernel_shifted_fft)
        temp2 = np.multiply(np.conjugate(p_shifted_fft), p_shifted_fft)
        for i in range(3):
            self.estimated_image_fft[:,:,i] = np.multiply(np.divide(np.conjugate(self.kernel_shifted_fft), (temp1+gamma_init*temp2)), self.blurred_image_shifted_fft[:,:,i])
            self.estimated_image[:,:,i] = np.fft.ifft2(np.fft.ifftshift(self.estimated_image_fft[:,:,i]))

        recovered_image = np.abs(self.estimated_image[0:self.image_size[0], 0:self.image_size[1], :])
        recovered_image = cv2.cvtColor(cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F), cv2.COLOR_BGR2RGB)

        psnr_list.append(mypsnr(self.original_image, recovered_image))
        ssim_list.append(inbuilt_ssim(self.original_image, recovered_image))
        parameter_list.append(gamma_init)

        plt.axis("off")
        recovered_image_plot = plt.imshow(recovered_image)
        plt.imsave('../plots-results/experiment-3/constrained_image_1.png', recovered_image)

        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
        gamma_slider = Slider(slider_ax, 'gamma', gamma_min, gamma_max, valinit=gamma_init)

        def update(gamma):
            for i in range(3):
                self.estimated_image_fft[:,:,i] = np.multiply(np.divide(np.conjugate(self.kernel_shifted_fft), (temp1+gamma*temp2)), self.blurred_image_shifted_fft[:,:,i])
            self.estimated_image[:,:,i] = np.fft.ifft2(np.fft.ifftshift(self.estimated_image_fft[:,:,i]))

            recovered_image = cv2.cvtColor(np.abs(self.estimated_image[0:self.image_size[0], 0:self.image_size[1], :]), cv2.COLOR_BGR2RGB)
            recovered_image = cv2.normalize(recovered_image, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
            psnr_list.append(mypsnr(self.original_image, recovered_image))
            ssim_list.append(inbuilt_ssim(self.original_image, recovered_image))
            parameter_list.append(gamma)
            recovered_image_plot.set_data(recovered_image)
            fig.canvas.draw_idle()

        gamma_slider.on_changed(update)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--functiontype', type=int, help='whichever function needs to be implemented')
    args = parser.parse_args()
    question = 1
    if args.functiontype is None or args.functiontype == 1:
        question = 1
    elif args.functiontype == 2:
        question = 2
    elif args.functiontype == 3:
        question = 3
    elif args.functiontype == 4:
        question = 4
    obj = imageRestoration(question)
    
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    parameter_list = np.array(parameter_list)

    plt.figure(1)
    plt.subplot(211);plt.grid(True);plt.xlabel("gamma");plt.ylabel("psnr");plt.plot(parameter_list, psnr_list)
    plt.subplot(212);plt.grid(True);plt.xlabel("gamma");plt.ylabel("ssim");plt.plot(parameter_list, ssim_list)
    plt.show()