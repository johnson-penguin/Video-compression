import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_psnr_map(img1, img2):
    # 計算每個像素的 MSE
    mse = (img1 - img2) ** 2
    mse = np.mean(mse, axis=2)  # 對 RGB 通道取平均

    # 計算 PSNR
    max_pixel = 255.0  # 對於8位圖像
    psnr_map = np.zeros(mse.shape)
    
    # 避免計算對於 MSE = 0 的情況
    with np.errstate(divide='ignore', invalid='ignore'):
        psnr_map = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # 將無限大值設為 255（或其他顏色）
    psnr_map[mse == 0] = 255

    return psnr_map

# 載入圖片
image1 = cv2.imread('penguin.jpg')
# image2 = cv2.imread('hdr_hlg_image.tiff')
image2 = cv2.imread('hdr_pq_image.tiff')

# 確保兩張圖片大小相同
if image1.shape == image2.shape:
    psnr_map = calculate_psnr_map(image1, image2)

    # 顯示 PSNR 地圖
    plt.imshow(psnr_map, cmap='jet')
    plt.colorbar(label='PSNR (dB)')
    plt.title('PSNR Map')
    plt.axis('on')  # 不顯示坐標軸
    plt.show()
else:
    print('Error: Images must have the same dimensions.')
