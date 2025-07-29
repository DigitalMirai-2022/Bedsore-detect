import cv2
import numpy as np
import os

def gamma_correction(img, gamma=1.0):
    """
    ガンマ補正を行う補助関数
    gamma=1.0より小さいと明るく、大きいと暗くなる
    """
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in range(256)]).astype("uint8")
    return cv2.LUT(img, look_up_table)

def detect_bedsores_edge(image_path, gamma_value=1.2, min_area=500):
    """
    画像からエッジ検出のみを用いて褥瘡らしき領域を抽出するサンプル関数

    Parameters
    ----------
    image_path : str
        画像ファイルパス
    gamma_value : float
        ガンマ補正値
    min_area : int
        検出する領域の最小面積
    Returns
    -------
    out_img : ndarray (BGR)
        入力画像に褥瘡候補領域の輪郭を描画した結果
    mask_final : ndarray (grayscale)
        抽出された褥瘡領域の二値マスク (0 or 255)
    """

    # 1. 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read the image: {image_path}")

    # 2. ガンマ補正（任意）
    img = gamma_correction(img, gamma=gamma_value)

    # 3. グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. エッジ検出（Canny法）の実行
    # ※閾値は画像の特性に応じて調整が必要です
    edges = cv2.Canny(gray, 20, 150)

    # 5. モルフォロジー演算でエッジの途切れを補完（閉じる処理）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6. 輪郭抽出して面積によるフィルタ
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(edges_closed)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # 小さい領域は除外
        # 面積が十分な領域を最終マスクに描画
        cv2.drawContours(mask_final, [cnt], -1, 255, thickness=-1)

    # 7. オリジナル画像に輪郭描画
    out_img = img.copy()
    contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_img, contours_final, -1, (0, 0, 255), 2)  # 輪郭を赤色で描画

    return out_img, mask_final

if __name__ == "__main__":
    import glob

    # 入力画像が置いてあるディレクトリ
    input_dir = "input_images"
    # 出力先ディレクトリ
    output_dir = "output_results_edge_only"
    # 必要ならばディレクトリを作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 画像ファイルの拡張子を指定してリストアップ（jpg, png など）
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files += glob.glob(os.path.join(input_dir, "*.png"))
    image_files += glob.glob(os.path.join(input_dir, "*.jpeg"))

    for img_path in image_files:
        print(f"Processing {img_path} ...")

        # detect_bedsores_edge関数を呼び出し（エッジ検出のみ）
        out_img, out_mask = detect_bedsores_edge(img_path)

        # ファイル名取得
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        # 結果画像の保存先
        out_img_path = os.path.join(output_dir, f"{name}_detect{ext}")
        out_mask_path = os.path.join(output_dir, f"{name}_mask{ext}")

        # 結果を保存
        cv2.imwrite(out_img_path, out_img)
        cv2.imwrite(out_mask_path, out_mask)

        print(f" -> Saved result: {out_img_path}, {out_mask_path}")
