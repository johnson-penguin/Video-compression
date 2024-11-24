import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def hlg_oetf(x):
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    return np.where(x <= 1/12, np.sqrt(3 * x), a * np.log(np.clip(12 * x - b, 1e-10, None)) + c)

# 讀取 SDR 圖像
sdr_image = cv2.imread('penguin.jpg', cv2.IMREAD_COLOR)
if sdr_image is None:
    raise FileNotFoundError("找不到圖像。請檢查文件路徑。")

# 歸一化到 [0, 1]
sdr_normalized = sdr_image.astype(np.float32) / 255.0
mean_luminance = np.mean(sdr_normalized)

# 調整增益
gain = np.clip(0.45 + 0.3 * (1 - mean_luminance), 0.5, 2.0)
max_luminance = 700.0  # 調低峰值亮度
hdr_linear = sdr_normalized * gain * (max_luminance / 100)

# 限制範圍映射
hdr_hlg_input = np.clip(hdr_linear / 12, 0.01, 0.99)
hdr_hlg = hlg_oetf(hdr_hlg_input)

# 增強對比度與亮度
contrast_factor = 1.2
brightness_offset = -0.05
hdr_hlg_adjusted = np.clip(hdr_hlg * contrast_factor + brightness_offset, 0, 1)

# 儲存為 16 位 TIFF
hdr_hlg_16bit = np.clip(hdr_hlg_adjusted * 65535, 0, 65535).astype(np.uint16)
cv2.imwrite('hdr_hlg_image_adjusted.tiff', hdr_hlg_16bit)

# 計算數值差異
difference = np.abs(sdr_normalized - hdr_hlg_adjusted)
mean_difference = np.mean(difference) * 100
max_difference = np.max(difference) * 100

print(f"平均差異百分比: {mean_difference:.2f}%")
print(f"最大差異百分比: {max_difference:.2f}%")

# SSIM 與 PSNR
hdr_hlg_image = cv2.imread('hdr_hlg_image_adjusted.tiff', cv2.IMREAD_UNCHANGED)
sdr_gray = cv2.cvtColor(sdr_image, cv2.COLOR_BGR2GRAY)
hdr_hlg_gray = cv2.cvtColor(hdr_hlg_image, cv2.COLOR_BGR2GRAY)

similarity_index = ssim(sdr_gray, hdr_hlg_gray)
psnr_value = psnr(sdr_gray, hdr_hlg_gray, data_range=hdr_hlg_gray.max() - hdr_hlg_gray.min())

print("SSIM:", similarity_index)
print("PSNR:", psnr_value)
