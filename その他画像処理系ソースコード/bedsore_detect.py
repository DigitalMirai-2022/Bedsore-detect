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

def detect_bedsores(image_path,
                    red_thresholds=None,
                    black_thresholds=None,
                    gamma_value=1.2,
                    min_area=500,
                    use_edge_detection=True):
    """
    画像から褥瘡らしき領域を抽出するサンプル関数

    Parameters
    ----------
    image_path : str
        画像ファイルパス
    red_thresholds : list of tuples
        赤系を抽出するためのHSV閾値リスト ([(Hmin, Smin, Vmin), (Hmax, Smax, Vmax)], ...)
    black_thresholds : tuple
        黒系を抽出するためのHSV閾値 ((Hmin, Smin, Vmin), (Hmax, Smax, Vmax))
    gamma_value : float
        ガンマ補正値
    min_area : int
        検出する領域の最小面積
    use_edge_detection : bool
        エッジ検出を併用するか否か
    Returns
    -------
    out_img : ndarray (BGR)
        入力画像に褥瘡候補領域の輪郭を描画した結果
    mask_final : ndarray (grayscale)
        抽出された褥瘡領域の二値マスク(0 or 255)
    """

    # 1. 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read the image: {image_path}")

    # 2. ガンマ補正（任意）
    img = gamma_correction(img, gamma=gamma_value)

    # 3. HSV色空間へ変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤系のデフォルトしきい値 (例)
    if red_thresholds is None:
        # 0-10度 と 170-180度あたりを赤色領域とみなす例
        red_thresholds = [
            ((0, 50, 30), (5, 255, 255)),
            ((150, 50, 30), (180, 255, 255))
        ]

    # 黒系のデフォルトしきい値 (例)
    if black_thresholds is None:
        # 画面上の暗部を抽出する例
        black_thresholds = ((0, 0, 0), (180, 255, 50))

    # 4. 赤色マスクの作成（複数閾値を OR で結合）
    mask_red_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in red_thresholds:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask_red = cv2.inRange(hsv, lower_np, upper_np)
        mask_red_total = cv2.bitwise_or(mask_red_total, mask_red)

    # 5. 黒色マスクの作成
    lower_black = np.array(black_thresholds[0], dtype=np.uint8)
    upper_black = np.array(black_thresholds[1], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 6. 褥瘡候補として赤系 or 黒系を OR で統合
    mask_combined = cv2.bitwise_or(mask_red_total, mask_black)

    # 7. モルフォロジー演算でノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_morph = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel, iterations=2)

    if use_edge_detection:
        # 8. エッジ検出（Canny法）の実行
        # グレースケール変換（エッジ検出にはグレースケール画像が一般的）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Canny法によるエッジ抽出（閾値は適宜調整が必要）
        edges = cv2.Canny(gray, 50, 150)

        # 9. 色ベースのマスクとエッジマップの統合
        # 例: エッジのある部分だけを残す（AND 演算）
        mask_morph = cv2.bitwise_and(mask_morph, edges)
        # 統合後に再度、モルフォロジー演算で穴埋めする（必要に応じて）
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 10. 輪郭抽出して面積によるフィルタ
    contours, hierarchy = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask_morph)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # 小さい領域は除外
        # 大きい領域を最終マスクに描画
        cv2.drawContours(mask_final, [cnt], -1, 255, thickness=-1)

    # 11. オリジナル画像に輪郭描画
    out_img = img.copy()
    contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_img, contours_final, -1, (0, 0, 255), 2)  # 赤色の枠で表示

    return out_img, mask_final

if __name__ == "__main__":
    import glob

    # 入力画像が置いてあるディレクトリ
    input_dir = "input_images"
    # 出力先ディレクトリ
    output_dir = "output_results_edge"
    # 必要ならばディレクトリを作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 画像ファイルの拡張子を指定してリストアップ（jpg, png など）
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files += glob.glob(os.path.join(input_dir, "*.png"))
    image_files += glob.glob(os.path.join(input_dir, "*.jpeg"))

    for img_path in image_files:
        print(f"Processing {img_path} ...")

        # detect_bedsores関数を呼び出し（エッジ検出も有効にする）
        out_img, out_mask = detect_bedsores(img_path, use_edge_detection=True)

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
