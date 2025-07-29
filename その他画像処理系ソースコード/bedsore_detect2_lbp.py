import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans


def detect_bedsore_by_texture_and_color(image_path, output_path, n_clusters=3):
    """
    Lab カラー空間 + LBP の特徴量を用いて K-means で褥瘡らしい領域を検出（簡易版）。
    n_clusters=3 程度に設定し、あとで褥瘡クラスタを推定する。
    """
    # 1. 画像読み込み
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {image_path}")

    # BGR → RGB 変換（可視化用）
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # BGR → Lab カラー空間変換
    # Lab空間は明度(L), 色度(a, b)に分かれており、皮膚などの微妙な色の違いを捉えやすい
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(img_lab)

    # グレースケール (LBP用)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. LBP を計算
    #   P=8, R=1, 'uniform' の設定は一例
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    # 0-255 に正規化
    lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-7) * 255

    # 3. カラー情報 + LBP を結合した特徴ベクトルを作成
    #   L, A, B, lbp_norm を 4次元特徴として使用
    #   shape: (h*w, 4)
    h, w = gray.shape
    L_flat = L.reshape(-1, 1).astype(np.float32)
    A_flat = A.reshape(-1, 1).astype(np.float32)
    B_flat = B.reshape(-1, 1).astype(np.float32)
    lbp_flat = lbp_norm.reshape(-1, 1).astype(np.float32)

    features = np.concatenate([L_flat, A_flat, B_flat, lbp_flat], axis=1)

    # 4. K-means クラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(features)
    labels = kmeans.labels_.reshape(h, w)

    # 5. どのクラスタが褥瘡か自動で決める（簡易版）
    #    - 各クラスタの平均 L 値や A/B 値、LBP 分散などを見て、
    #      「比較的赤黒っぽい or LBP が大きい」クラスタを褥瘡とみなす等のヒューリスティックが考えられる
    #    - ここでは例として、クラスタごとの平均 A (赤緑) + B (黄青) 値が特定範囲に近いものを褥瘡と仮定
    #      実際は、褥瘡画像の統計をとってルール化する必要がある

    # クラスタごとの (A,B) の平均を計算
    cluster_info = {}
    for c in range(n_clusters):
        mask_c = (labels == c)
        A_mean = A[mask_c].mean() if np.sum(mask_c) > 0 else 0
        B_mean = B[mask_c].mean() if np.sum(mask_c) > 0 else 0
        cluster_info[c] = (A_mean, B_mean)

    # 例: 赤っぽい（Aが大きい）、かつ B がそこそこなクラスタを褥瘡と仮定する
    # 実際には黒色・黄色・白色など褥瘡の色は多様なので、もっと複雑なルールが必要
    # ここでは "A_mean が最大のクラスタ" を褥瘡とする、という簡単な例
    sore_cluster = max(cluster_info, key=lambda c: cluster_info[c][0])

    # 6. 褥瘡と推定したクラスタのマスクを作成
    bed_sore_mask = (labels == sore_cluster).astype(np.uint8)

    # 7. マスクの後処理（ノイズ除去や穴埋めなど）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bed_sore_mask = cv2.morphologyEx(bed_sore_mask, cv2.MORPH_CLOSE, kernel)
    bed_sore_mask = cv2.morphologyEx(bed_sore_mask, cv2.MORPH_OPEN, kernel)

    # 8. 結果の可視化 (赤マスク重ね表示)
    result_img = img_rgb.copy()
    result_img[bed_sore_mask == 1] = [255, 0, 0]  # 赤

    # matplotlib で表示＆保存
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img_rgb)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(result_img)
    axs[1].set_title('Detection Result (Red=Detected Cluster)')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    input_dir = "input_images"
    output_dir = "output_results_lbp"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + "_lbp_color_result.png")

            try:
                detect_bedsore_by_texture_and_color(image_path, output_path, n_clusters=3)
                print(f"{filename} の解析が完了し、結果を保存しました: {output_path}")
            except Exception as e:
                print(f"{filename} の解析中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
