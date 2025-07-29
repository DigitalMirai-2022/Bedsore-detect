import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread("input_images/toko23.JPG")
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# スライダーのコールバック関数（値を取得するだけ）
def nothing(x):
    pass


# ウィンドウ作成
cv2.namedWindow("Trackbars")

# スライダーを作成 (H=0-180, S=0-255, V=0-255)
cv2.createTrackbar("H_min", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("H_max", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)

while True:
    # スライダーの値を取得
    h_min = cv2.getTrackbarPos("H_min", "Trackbars")
    h_max = cv2.getTrackbarPos("H_max", "Trackbars")
    s_min = cv2.getTrackbarPos("S_min", "Trackbars")
    s_max = cv2.getTrackbarPos("S_max", "Trackbars")
    v_min = cv2.getTrackbarPos("V_min", "Trackbars")
    v_max = cv2.getTrackbarPos("V_max", "Trackbars")

    # 指定範囲のマスク作成
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(image_hsv, lower, upper)

    # 元画像とマスクのAND演算（赤色部分を強調）
    result = cv2.bitwise_and(image, image, mask=mask)

    # 画像表示
    cv2.imshow("Original", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
