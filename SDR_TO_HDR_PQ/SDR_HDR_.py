import cv2
import numpy as np

def pq_eotf(x):
    """PQ EOTF (Electro-Optical Transfer Function)"""
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    
    x_pow = np.power(x, 1/m2)
    num = np.maximum(x_pow - c1, 0)
    den = c2 - c3 * x_pow
    return 10000 * np.power(num / den, 1/m1)

def pq_oetf(x):
    """PQ OETF (Opto-Electronic Transfer Function)"""
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    
    x = x / 10000  # Normalize to 0-1 range
    num = c1 + c2 * np.power(x, m1)
    den = 1 + c3 * np.power(x, m1)
    return np.power(num / den, m2)

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

# 自適應伽瑪值
gamma = 2.4  # 使用標準 sRGB 伽瑪值
max_luminance = 1000.0  # HDR 的最大亮度 (nits)

# 應用伽瑪校正和亮度映射
hdr_linear = np.power(sdr_normalized, gamma) * max_luminance

# 應用 PQ OETF
hdr_pq = pq_oetf(hdr_linear)

# 將 PQ 值縮放到 0-65535 範圍（16位）
hdr_pq_16bit = np.clip(hdr_pq * 65535, 0, 65535).astype(np.uint16)

# 保存為 16 位 TIFF 文件
cv2.imwrite('hdr_pq_image.tiff', hdr_pq_16bit)

# 讀取保存的 HDR PQ 圖像
hdr_pq_image = cv2.imread('hdr_pq_image.tiff', cv2.IMREAD_UNCHANGED)

# 檢查圖像是否正確加載
if hdr_pq_image is None:
    raise FileNotFoundError("找不到 HDR PQ 圖像。請檢查文件路徑。")

# 檢查圖像數據類型和深度
print("\n轉換後的 HDR PQ 圖像:")
print("Data Type:", hdr_pq_image.dtype)
print("Shape:", hdr_pq_image.shape)

print("\n轉換完成。HDR PQ 圖像已保存為 'hdr_pq_image.tiff'")