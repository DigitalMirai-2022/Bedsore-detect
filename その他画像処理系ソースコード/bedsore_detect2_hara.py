import cv2
import numpy as np
import os
import glob

from skimage.feature import graycomatrix, graycoprops

def gamma_correction(img, gamma=1.0):
    """
    ガンマ補正を行う補助関数
    gamma < 1.0 なら明るく、gamma > 1.0 なら暗くなる
    """
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in range(256)]).astype("uint8")
    return cv2.LUT(img, look_up_table)

def detect_bedsores_texture(image_path, gamma_value=1.2, min_area=500):
    """
    画像からHaralick特徴量（GLCM：コントラスト）を用いて
    褥瘡候補領域を抽出するサンプル関数
    """
    # 1. 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read the image: {image_path}")

    # 2. ガンマ補正（任意）
    img = gamma_correction(img, gamma=gamma_value)

    # 3. グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ##### Haralick特徴量によるテクスチャ解析（GLCM：コントラスト） #####
    # 例: 窓サイズを31に設定
    ws = 31
    h, w = gray.shape
    contrast_map = np.zeros_like(gray, dtype=np.float32)

    # 画素をウィンドウ単位で走査してGLCMからコントラストを算出
    for i in range(0, h - ws, ws):
        for j in range(0, w - ws, ws):
            window = gray[i:i+ws, j:j+ws]
            glcm = graycomatrix(window, distances=[1],
                                 angles=[0],
                                 levels=256,
                                 symmetric=True,
                                 normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            contrast_map[i:i+ws, j:j+ws] = contrast

    # コントラストマップを正規化して0-255の範囲にスケーリング
    contrast_map_norm = np.uint8(
        255 * (contrast_map - contrast_map.min()) /
        (contrast_map.max() - contrast_map.min() + 1e-7)
    )

    # Otsuの手法による二値化
    _, haralick_mask = cv2.threshold(contrast_map_norm, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ##### マスクの後処理 #####
    # モルフォロジー処理で小さな穴や隙間を補完（閉じる処理）
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    haralick_mask = cv2.morphologyEx(haralick_mask, cv2.MORPH_CLOSE, kernel_morph, iterations=1)

    # 小さなノイズ領域の除去（面積フィルタ）
    contours, _ = cv2.findContours(haralick_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(haralick_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cv2.drawContours(mask_final, [cnt], -1, 255, thickness=-1)

    # オリジナル画像に最終候補領域の輪郭を描画
    out_img = img.copy()
    contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_img, contours_final, -1, (0, 0, 255), 2)  # 輪郭を赤色で描画

    return out_img, mask_final

if __name__ == "__main__":
    input_dir = "input_images"
    output_dir = "output_results_hara"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files += glob.glob(os.path.join(input_dir, "*.png"))
    image_files += glob.glob(os.path.join(input_dir, "*.jpeg"))

    for img_path in image_files:
        print(f"Processing {img_path} ...")

        out_img, out_mask = detect_bedsores_texture(img_path,
                                                    gamma_value=1.2,
                                                    min_area=500)

        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        out_img_path = os.path.join(output_dir, f"{name}_detect{ext}")
        out_mask_path = os.path.join(output_dir, f"{name}_mask{ext}")

        cv2.imwrite(out_img_path, out_img)
        cv2.imwrite(out_mask_path, out_mask)

        print(f" -> Saved result: {out_img_path}, {out_mask_path}")
