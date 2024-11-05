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

def calculate_color_ssim(A_color, B_color):
    """Calculate the SSIM and SSIM map for each RGB channel and average the results."""
    ssim_r, ssim_map_r = ssim(A_color[:, :, 0], B_color[:, :, 0])
    ssim_g, ssim_map_g = ssim(A_color[:, :, 1], B_color[:, :, 1])
    ssim_b, ssim_map_b = ssim(A_color[:, :, 2], B_color[:, :, 2])

    # Combine the three SSIM maps into a single color SSIM map
    ssim_map_color = np.stack((ssim_map_r, ssim_map_g, ssim_map_b), axis=-1)
    color_ssim = (ssim_r + ssim_g + ssim_b) / 3
    return color_ssim, ssim_map_color



def main():
    A_color = cv2.imread('penguin.jpg', cv2.IMREAD_COLOR)
    B_color = cv2.imread('hdr_hlg_image.tiff', cv2.IMREAD_COLOR)

    if A_color.shape != B_color.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # Calculate SSIM for color images
    color_ssim_value, ssim_map_color = calculate_color_ssim(A_color, B_color)
    print(f'SSIM (Color): {color_ssim_value}')

    # Convert BGR to RGB for displaying
    A_rgb = cv2.cvtColor(A_color, cv2.COLOR_BGR2RGB)
    B_rgb = cv2.cvtColor(B_color, cv2.COLOR_BGR2RGB)
    ssim_map_rgb = (ssim_map_color * 255).astype(np.uint8)  # Scale SSIM map for display

    # Display both images and SSIM value
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'SSIM (Color): {color_ssim_value:.4f}', fontsize=16)

    plt.subplot(1, 3, 1)
    plt.title('Image A')
    plt.imshow(A_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Image B')
    plt.imshow(B_rgb)
    plt.axis('off')


    plt.subplot(1, 3, 3)
    plt.title('SSIM Map')
    plt.imshow(np.mean(ssim_map_color, axis=-1), cmap='hot_r', vmin=0, vmax=1)  # 使用反轉的 colormap

    plt.colorbar()
    plt.axis('off')


    plt.show()

if __name__ == "__main__":
    main()