import cv2
import numpy as np

def hlg_oetf(x):
    """HLG OETF (Opto-Electronic Transfer Function)"""
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    return np.where(x <= 1/12, np.sqrt(3 * x), a * np.log(12 * x - b) + c)

def hlg_inverse_oetf(x):
    """HLG inverse OETF"""
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    return np.where(x <= 0.5, x**2 / 3, np.exp((x - c) / a) / 12 + b / 12)

# 讀取 SDR 圖像
sdr_image = cv2.imread('penguin.jpg', cv2.IMREAD_COLOR)

# 檢查圖像是否正確加載
if sdr_image is None:
    raise FileNotFoundError("找不到圖像。請檢查文件路徑。")

# 檢查圖像數據類型和深度
print("原始 SDR 圖像:")
print("Data Type:", sdr_image.dtype)
print("Shape:", sdr_image.shape)

# 將 SDR 圖像轉換為浮點數並歸一化到 [0, 1] 範圍
sdr_normalized = sdr_image.astype(np.float32) / 255.0

# 計算圖像的平均亮度
mean_luminance = np.mean(sdr_normalized)

# 自適應增益調整
gain = 1.2 + 0.8 * (1 - mean_luminance)  # 根據平均亮度調整增益
max_luminance = 1000.0  # HDR 的最大亮度 (nits)

# 應用 sRGB 逆伽瑪校正（假設輸入是 sRGB）
sdr_linear = np.power(sdr_normalized, 2.2)

# 應用增益調整
hdr_linear = sdr_linear * gain * (max_luminance / 100)  # 假設 SDR 峰值亮度為 100 nits

# 應用 HLG OETF
hdr_hlg = hlg_oetf(hdr_linear / 12)  # HLG 通常將輸入標準化到 0-12 範圍

# 將 HLG 值縮放到 0-65535 範圍（16位）
hdr_hlg_16bit = np.clip(hdr_hlg * 65535, 0, 65535).astype(np.uint16)

# 保存為 16 位 TIFF 文件
cv2.imwrite('hdr_hlg_image.tiff', hdr_hlg_16bit)

# 讀取保存的 HDR HLG 圖像
hdr_hlg_image = cv2.imread('hdr_hlg_image.tiff', cv2.IMREAD_UNCHANGED)

# 檢查圖像是否正確加載
if hdr_hlg_image is None:
    raise FileNotFoundError("找不到 HDR HLG 圖像。請檢查文件路徑。")

# 檢查圖像數據類型和深度
print("\n轉換後的 HDR HLG 圖像:")
print("Data Type:", hdr_hlg_image.dtype)
print("Shape:", hdr_hlg_image.shape)

print("\n轉換完成。HDR HLG 圖像已保存為 'hdr_hlg_image.tiff'")