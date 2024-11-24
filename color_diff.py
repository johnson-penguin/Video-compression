import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def calculate_region_differences(img1, img2, region_size=32, diff_threshold=20):
    """
    計算每個區域的顯著差異
    Parameters:
        img1: 第一張圖片
        img2: 第二張圖片
        region_size: 區域大小 (預設為 32x32)
        diff_threshold: 判定顯著差異的閾值 (預設為 20)
    Returns:
        diff_regions: 差異值列表
        positions: 顯著差異區域的左上角座標
        total_regions: 總區域數量
    """
    h, w = img1.shape[:2]
    diff_regions = []
    positions = []
    total_regions = 0
    
    for y in range(0, h, region_size):
        for x in range(0, w, region_size):
            total_regions += 1
            # 提取區域
            region1 = img1[y:y+region_size, x:x+region_size]
            region2 = img2[y:y+region_size, x:x+region_size]
            # 計算區域差異 (平均絕對差)
            diff = np.mean(np.abs(region1 - region2))
            if diff > diff_threshold:  # 判定是否超過顯著差異閾值
                diff_regions.append(diff)
                positions.append((x, y))
    
    return diff_regions, positions, total_regions

def analyze_image_colors(img1_path, img2_path, region_size=32, diff_threshold=20):
    """
    深入分析兩張圖片的顏色分布差異並計算相異百分比
    Parameters:
        img1_path: 第一張圖片路徑
        img2_path: 第二張圖片路徑
        region_size: 區域大小 (預設為 32x32)
        diff_threshold: 判定顯著差異的閾值 (預設為 20)
    Returns:
        None (顯示分析結果)
    """
    # 讀取圖片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 確保圖片大小相同
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 轉換到 RGB 色彩空間進行分析
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 計算區域差異統計
    diff_regions, positions, total_regions = calculate_region_differences(img1_rgb, img2_rgb, region_size, diff_threshold)
    num_diff_regions = len(diff_regions)
    percent_difference = (num_diff_regions / total_regions) * 100

    # 顯示區域差異分析結果
    print("\n區域差異分析:")
    print(f"總區域數量: {total_regions}")
    print(f"顯著差異區域數量: {num_diff_regions}")
    print(f"相異百分比: {percent_difference:.2f}%")
    
    if num_diff_regions > 0:
        print(f"最大區域差異值: {max(diff_regions):.2f}")
        print("差異最大的區域位置：", positions[diff_regions.index(max(diff_regions))])

    # 顯示差異圖
    rgb_diff = cv2.absdiff(img1_rgb, img2_rgb)
    diff_rgb_display = np.sum(rgb_diff, axis=2)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(131)
    plt.title("Original Image 1")
    plt.imshow(img1_rgb)
    plt.axis('off')

    plt.subplot(132)
    plt.title("Original Image 2")
    plt.imshow(img2_rgb)
    plt.axis('off')

    plt.subplot(133)
    plt.title("RGB Difference")
    plt.imshow(diff_rgb_display, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # 使用示例
    img1_path = "penguin.jpg"   # 替換為您的第一張圖片路徑
    img2_path = "hdr_hlg_image.tiff"  # 替換為您的第二張圖片路徑
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("請確保圖片路徑正確")
        return
    
    # 設定區域大小和差異閾值
    region_size = 128
    diff_threshold = 20

    analyze_image_colors(img1_path, img2_path, region_size, diff_threshold)

if __name__ == "__main__":
    main()
