import cv2
import numpy as np
import os
import glob

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

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
    画像からテクスチャ解析（LBP, Gaborフィルタ, Haralick特徴量）を用いて
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

    ##### ① LBPによるテクスチャ解析 (パラメータ調整) #####
    # 例: P=16, R=2 に変更
    P = 16
    R = 2
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    # 正規化
    lbp_norm = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-7))

    # --- Otsuによる二値化 ---
    _, lbp_mask_otsu = cv2.threshold(lbp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # --- あるいは自前の閾値（例: 平均+標準偏差を使うなど）---
    # mean_val, std_val = cv2.meanStdDev(lbp_norm)
    # custom_thresh = mean_val[0][0] + std_val[0][0] * 0.5  # 適当な係数
    # _, lbp_mask_custom = cv2.threshold(lbp_norm, custom_thresh, 255, cv2.THRESH_BINARY)
    lbp_mask = lbp_mask_otsu  # ここで使うマスクを決定

    ##### ② Gaborフィルタによるテクスチャ解析 (パラメータ調整) #####
    # 例: ksize=31, sigma=8.0, lambd=15.0, gamma_param=0.7 に設定
    ksize = 31
    sigma = 8.0
    lambd = 15.0
    gamma_param = 0.7
    psi = 0
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    gabor_responses = []
    for theta in angles:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd,
                                    gamma_param, psi, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        gabor_responses.append(filtered)

    gabor_max = np.max(np.stack(gabor_responses, axis=-1), axis=-1)
    gabor_max_norm = cv2.normalize(gabor_max, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- Otsuによる二値化 ---
    _, gabor_mask_otsu = cv2.threshold(gabor_max_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # --- あるいは自前の閾値 ---
    # mean_val2, std_val2 = cv2.meanStdDev(gabor_max_norm)
    # custom_thresh2 = mean_val2[0][0] + std_val2[0][0] * 0.5
    # _, gabor_mask_custom = cv2.threshold(gabor_max_norm, custom_thresh2, 255, cv2.THRESH_BINARY)
    gabor_mask = gabor_mask_otsu  # ここで使うマスクを決定

    ##### ③ Haralick特徴量（GLCM：コントラスト） (パラメータ調整) #####
    # 例: 窓サイズを31に拡大
    ws = 31
    h, w = gray.shape
    contrast_map = np.zeros_like(gray, dtype=np.float32)

    # 角度0°のみだが、必要に応じて複数角度の平均を取るのもアリ
    for i in range(0, h - ws, ws):
        for j in range(0, w - ws, ws):
            window = gray[i:i+ws, j:j+ws]
            glcm = graycomatrix(window, distances=[1],
                                angles=[0], levels=256,
                                symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            contrast_map[i:i+ws, j:j+ws] = contrast

    contrast_map_norm = np.uint8(
        255 * (contrast_map - contrast_map.min()) /
        (contrast_map.max() - contrast_map.min() + 1e-7)
    )

    # --- Otsuによる二値化 ---
    _, haralick_mask_otsu = cv2.threshold(contrast_map_norm, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # --- あるいは自前の閾値 ---
    # mean_val3, std_val3 = cv2.meanStdDev(contrast_map_norm)
    # custom_thresh3 = mean_val3[0][0] + std_val3[0][0] * 0.5
    # _, haralick_mask_custom = cv2.threshold(contrast_map_norm,
    #                                         custom_thresh3, 255,
    #                                         cv2.THRESH_BINARY)a
    haralick_mask = haralick_mask_otsu  # ここで使うマスクを決定

    ##### ④ マスクの結合と後処理 #####
    # ■ ORで単純に足し合わせると過検出になりやすいので、ANDや他の論理で絞る方法もある
    # (例) 3つのうち2つ以上が白なら採用するロジック → 今回は簡易サンプルで AND を例示

    # AND演算で全てが白な領域のみ使う（かなり厳しい判定）
    combined_mask = cv2.bitwise_or(lbp_mask, gabor_mask)
    combined_mask = cv2.bitwise_or(combined_mask, haralick_mask)

    # もし過検出気味なら AND を使い、逆に見落としが多ければ OR を使い、状況に応じて調整してください。

    # モルフォロジー処理で小さな穴や隙間を補完（閉じる処理）
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_morph, iterations=1)

    # 小さなノイズ領域の除去（面積フィルタ）
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(combined_mask)
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
    output_dir = "output_results_texture"
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


#
