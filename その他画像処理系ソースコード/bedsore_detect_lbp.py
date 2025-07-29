import cv2
import numpy as np
import matplotlib.pyplot as plt

# scikit-image からLBPを計算する関数をインポート
from skimage.feature import local_binary_pattern

# K-meansクラスタリングを使用
from sklearn.cluster import KMeans


def detect_bedsore_by_texture(image_path):
    """
    LBP（Local Binary Pattern）を用いた簡易的なテクスチャ解析＋K-meansによる褥瘡検出デモ
    """
    # 1. 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {image_path}")

    # BGR → RGB 変換（表示用）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. LBP計算
    #   - P=8近傍, R=1ピクセル半径, 'uniform'は代表的なLBPバリエーション
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')

    # LBPの値は浮動小数になるので、画像として扱うために正規化しておく（0-255スケール）
    lbp_normalized = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-7))

    # 3. K-meansクラスタリングによるセグメンテーション
    #   - ここでは画素単位でLBP値をクラスタリングにかける簡易的手法
    #   - 特徴量としてLBPの値を使用（より多次元の特徴量を使うと精度向上が期待できます）
    h, w = gray.shape
    lbp_flat = lbp_normalized.reshape(-1, 1).astype(np.float32)  # (画素数, 1)

    # K-means実行
    # cluster数=2: 褥瘡領域(1)と正常皮膚領域(0)に大まかに分ける想定
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(lbp_flat)
    labels = kmeans.labels_

    # 4. 結果の整形
    #   - labelsを画像サイズにリシェイプ
    labels_img = labels.reshape(h, w)

    # K-meansのラベルは0/1どちらが褥瘡かは自動ではわからない
    # 一般的には、LBP値が大きく変化している部分を褥瘡とみなすなど、
    # 画像に応じて判断する必要がある
    # ここでは、単純にラベル1側を褥瘡とみなす例を示す

    # もし、どちらが褥瘡か自動判定したい場合は、LBP値の平均や輪郭の形状など
    # 追加のヒューリスティックを加えて判定する
    bed_sore_label = 1  # 仮定
    bed_sore_mask = (labels_img == bed_sore_label).astype(np.uint8)

    # 5. マスクの後処理（例: クロージングでノイズ除去）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bed_sore_mask = cv2.morphologyEx(bed_sore_mask, cv2.MORPH_CLOSE, kernel)

    # 6. 結果の可視化
    #   - マスク部分を赤色で重ね合わせて表示
    result_img = img_rgb.copy()
    # マスクが1の部分だけ赤を上書き
    result_img[bed_sore_mask == 1] = [255, 0, 0]  # 赤

    # matplotlib で表示
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_rgb)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(lbp_normalized, cmap='gray')
    axs[1].set_title('LBP Image')
    axs[1].axis('off')

    axs[2].imshow(result_img)
    axs[2].set_title('Detection Result (Red=Detected)')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    return bed_sore_mask


# 実行例（画像パスを指定して呼び出す）
if __name__ == "__main__":
    # 例: detect_bedsore_by_texture("bedsore_example.jpg")
    # 実際の褥瘡画像のパスに置き換えて使用してください
    pass
