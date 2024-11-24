import cv2
import numpy as np

def calculate_peak_luminance_in_nits(image_path, max_nits=100):
    """
    計算圖片的峰值亮度並轉換為 nits 單位。

    :param image_path: 圖片的路徑
    :param max_nits: 最大亮度範圍（默認為 SDR 的 100 nits）
    :return: 峰值亮度 (nits)
    """
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"無法載入圖片: {image_path}")

    # 將圖片從 BGR 轉為 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 正規化到 [0, 1]
    img_normalized = img_rgb / 255.0

    # 計算線性亮度（Luminance）: Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    luminance = 0.2126 * img_normalized[:, :, 0] + \
                0.7152 * img_normalized[:, :, 1] + \
                0.0722 * img_normalized[:, :, 2]

    # 找到峰值亮度
    peak_luminance_normalized = np.max(luminance)

    # 轉換為 nits
    peak_luminance_nits = peak_luminance_normalized * max_nits

    return peak_luminance_nits

# 使用範例
image_path = "penguin.jpg"  # 替換為您的圖片路徑
try:
    peak_luminance = calculate_peak_luminance_in_nits(image_path, max_nits=100)
    print(f"圖片的峰值亮度為: {peak_luminance:.2f} nits")
except FileNotFoundError as e:
    print(e)
