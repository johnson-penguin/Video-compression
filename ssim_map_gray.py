import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1.5):
    """Generate a Gaussian kernel."""
    k = cv2.getGaussianKernel(size, sigma)
    kernel = k * k.T
    return kernel

def filter2d(img, kernel):
    """Apply 2D filter to an image."""
    return cv2.filter2D(img, -1, kernel)

def ssim(A, B, c1=1/math.sqrt(255), c2=1/math.sqrt(255),l=255):
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    kernel = gaussian_kernel(22, 3)

    muA = filter2d(A, kernel)
    muB = filter2d(B, kernel)
    
    muA_sq = muA * muA
    muB_sq = muB * muB
    muA_muB = muA * muB
    
    sigmaA_sq = filter2d(A * A, kernel) - muA_sq
    sigmaB_sq = filter2d(B * B, kernel) - muB_sq
    sigmaAB = filter2d(A * B, kernel) - muA_muB
    
    ssim_map = ((2 * muA_muB + (c1*l)*(c1*l)) * (2 * sigmaAB + (c2*l)*(c2*l))) / ((muA_sq + muB_sq + (c1*l)*(c1*l)) * (sigmaA_sq + sigmaB_sq + (c2*l)*(c2*l)))
    return ssim_map.mean(), ssim_map

def main():
    A = cv2.imread('penguin.jpg', cv2.IMREAD_GRAYSCALE)
    # B = cv2.imread('hdr_hlg_image.tiff', cv2.IMREAD_GRAYSCALE)
    B = cv2.imread('hdr_pq_image.tiff', cv2.IMREAD_GRAYSCALE)

    if A.shape != B.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    ssim_value, ssim_map = ssim(A, B)
    print(f'SSIM: {ssim_value}')

    plt.figure(figsize=(10, 5))
    plt.suptitle(f'SSIM: {ssim_value:.4f}', fontsize=16)

    plt.subplot(1, 3, 1)
    plt.title('Image A')
    plt.imshow(A, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Image B')
    plt.imshow(B, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('SSIM Map')
    plt.imshow(ssim_map, cmap='hot')
    plt.colorbar()
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
