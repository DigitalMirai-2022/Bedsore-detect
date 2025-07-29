import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# ===== カメラ内部パラメータ（例：適宜調整してください） =====
FOCAL_LENGTH = 3  # 単位: mm（例：スマホカメラの場合）  # 焦点距離（要設定）   # 元4.15
SENSOR_WIDTH = 3.02  # 単位: mm（例：スマホカメラの場合）  # センサーサイズ（要設定） # 元6.17
# https://www.strapya.com/blogs/hameefun/13771?srsltid=AfmBOop963LNVecSGjHoY_8rsnBGGckdLLlHAFY6SwDnaFpYtVF_47nl


# ===== 褥瘡検出に関する関数（既存実装） =====
def gamma_correction(img, gamma=1.0):
    """
    ガンマ補正を行う補助関数
    gamma < 1.0 なら明るく、gamma > 1.0 なら暗くなる
    """
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in range(256)]).astype("uint8")
    return cv2.LUT(img, look_up_table)


def detect_bedsores(image_path,
                    red_thresholds=None,
                    black_thresholds=None,
                    gamma_value=1.2,
                    min_area=500):
    """
    画像から褥瘡らしき領域を抽出するサンプル関数
    """
    # 1. 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read the image: {image_path}")

    # 2. ガンマ補正
    img = gamma_correction(img, gamma=gamma_value)

    # 3. HSV色空間へ変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4. 赤色領域の抽出（複数閾値をORで結合）
    if red_thresholds is None:
        red_thresholds = [
            ((0, 50, 30), (5, 255, 255)),
            ((150, 50, 30), (180, 255, 255))
        ]
    mask_red_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in red_thresholds:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask_red = cv2.inRange(hsv, lower_np, upper_np)
        mask_red_total = cv2.bitwise_or(mask_red_total, mask_red)

    # 5. 黒色領域の抽出
    if black_thresholds is None:
        black_thresholds = ((0, 0, 0), (180, 100, 100))  # ★元(180, 255, 50)
    lower_black = np.array(black_thresholds[0], dtype=np.uint8)
    upper_black = np.array(black_thresholds[1], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 6. 赤または黒を統合
    mask_combined = cv2.bitwise_or(mask_red_total, mask_black)

    # 7. モルフォロジー演算でノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_morph = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 8. 輪郭抽出と面積フィルタ
    contours, hierarchy = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask_morph)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cv2.drawContours(mask_final, [cnt], -1, 255, thickness=-1)

    # 9. 結果画像に輪郭描画（赤色）
    out_img = img.copy()
    contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_img, contours_final, -1, (0, 0, 255), 2)

    return out_img, mask_final


# ===== ドラッグ可能な頂点クラス（ヒット領域付き・ドラッグ反応改善） =====
class DraggablePoint:
    def __init__(self, canvas, x, y, vis_radius=5, hit_radius=5, fill="blue", callback=None):
        self.canvas = canvas
        self.callback = callback  # 移動時に呼ばれるコールバック関数
        self.x = x
        self.y = y
        self.vis_radius = vis_radius
        self.hit_radius = hit_radius
        # ヒット領域（見た目は表示しない）
        self.hit_id = self.canvas.create_oval(
            x - hit_radius, y - hit_radius, x + hit_radius, y + hit_radius,
            fill="", outline=""
        )
        # 可視用の円
        self.vis_id = self.canvas.create_oval(
            x - vis_radius, y - vis_radius, x + vis_radius, y + vis_radius,
            fill=fill, outline=fill
        )
        # イベントバインド
        for obj_id in (self.hit_id, self.vis_id):
            self.canvas.tag_bind(obj_id, "<ButtonPress-1>", self.on_press)
            self.canvas.tag_bind(obj_id, "<B1-Motion>", self.on_motion)
            self.canvas.tag_bind(obj_id, "<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        self.offset_x = self.x - event.x
        self.offset_y = self.y - event.y
        self.canvas.config(cursor="hand2")

    def on_motion(self, event):
        new_x = event.x + self.offset_x
        new_y = event.y + self.offset_y
        self.x = new_x
        self.y = new_y
        self.canvas.coords(self.hit_id,
                           new_x - self.hit_radius, new_y - self.hit_radius,
                           new_x + self.hit_radius, new_y + self.hit_radius)
        self.canvas.coords(self.vis_id,
                           new_x - self.vis_radius, new_y - self.vis_radius,
                           new_x + self.vis_radius, new_y + self.vis_radius)
        if self.callback is not None:
            self.callback()

    def on_release(self, event):
        self.canvas.config(cursor="arrow")

    def disable(self):
        for obj_id in (self.hit_id, self.vis_id):
            self.canvas.tag_unbind(obj_id, "<ButtonPress-1>")
            self.canvas.tag_unbind(obj_id, "<B1-Motion>")
            self.canvas.tag_unbind(obj_id, "<ButtonRelease-1>")

    def enable(self):
        for obj_id in (self.hit_id, self.vis_id):
            self.canvas.tag_bind(obj_id, "<ButtonPress-1>", self.on_press)
            self.canvas.tag_bind(obj_id, "<B1-Motion>", self.on_motion)
            self.canvas.tag_bind(obj_id, "<ButtonRelease-1>", self.on_release)


# ===== 補助関数：閉曲線上に均等に点をサンプリング =====
def get_evenly_spaced_points(pts, num_points=8):
    pts = np.squeeze(pts)
    if pts.ndim == 1:
        pts = np.array([pts])

    dists = []
    for i in range(len(pts)):
        pt1 = pts[i]
        pt2 = pts[(i + 1) % len(pts)]
        d = np.linalg.norm(pt2 - pt1)
        dists.append(d)
    dists = np.array(dists)
    total_length = np.sum(dists)

    target_dists = np.linspace(0, total_length, num_points + 1)[:-1]
    sampled_points = []
    cumulative = 0
    seg_idx = 0
    for td in target_dists:
        while cumulative + dists[seg_idx] < td:
            cumulative += dists[seg_idx]
            seg_idx = (seg_idx + 1) % len(pts)
        remaining = td - cumulative
        ratio = remaining / dists[seg_idx] if dists[seg_idx] != 0 else 0
        pt1 = pts[seg_idx]
        pt2 = pts[(seg_idx + 1) % len(pts)]
        interp = pt1 + ratio * (pt2 - pt1)
        sampled_points.append((float(interp[0]), float(interp[1])))
    return sampled_points


# ===== 補助関数：多角形の面積をシューレースの公式で計算 =====
def polygon_area(points):
    """points: [(x1, y1), (x2, y2), ..., (xn, yn)]"""
    area = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2


# ===== メインGUIアプリ =====
class BedsoresDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("褥瘡検出・修正ツール")
        # ウィンドウサイズ固定
        self.canvas_width = 600
        self.canvas_height = 400
        self.master.resizable(False, False)

        # 画像・検出結果関連
        self.image_path = None
        self.original_cv_img = None  # 元画像（OpenCV形式, 元解像度）
        self.detected_img = None  # 検出後画像（元解像度）
        self.mask_final = None  # 検出マスク（元解像度）
        self.display_scale = 1.0
        self.display_width = self.canvas_width
        self.display_height = self.canvas_height
        self.offset_x = 0
        self.offset_y = 0

        # 八角形・頂点関連
        self.draggable_points = []
        self.octagon_id = None
        self.octagon_confirmed = False  # 「確定」状態かどうか

        # UIレイアウト
        self.btn_frame = tk.Frame(self.master)
        self.btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 修正モード表示ラベル（初期状態は非表示）
        self.mode_label = tk.Label(self.btn_frame, text="修正モード", fg="red", font=("Arial", 14))
        self.mode_label.pack_forget()

        self.upload_btn = tk.Button(self.btn_frame, text="画像アップロード", command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        self.detect_btn = tk.Button(self.btn_frame, text="検出", command=self.detect, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        self.modify_btn = tk.Button(self.btn_frame, text="修正", command=self.modify, state=tk.DISABLED)
        self.modify_btn.pack(side=tk.LEFT, padx=5)

        self.confirm_btn = tk.Button(self.btn_frame, text="確定", command=self.confirm, state=tk.DISABLED)
        self.confirm_btn.pack(side=tk.LEFT, padx=5)

        # 新たに「計測」ボタンを追加（初期は無効）
        self.measure_btn = tk.Button(self.btn_frame, text="計測", command=self.measure, state=tk.DISABLED)
        self.measure_btn.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="gray")
        self.canvas.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="画像を選択",
                                               filetypes=[("Image Files", "*.jpg;*.png;*.jpeg"), ("All Files", "*.*")])
        if not file_path:
            return
        self.image_path = file_path

        self.original_cv_img = cv2.imread(self.image_path)
        if self.original_cv_img is None:
            messagebox.showerror("エラー", "画像の読み込みに失敗しました。")
            return

        orig_h, orig_w = self.original_cv_img.shape[:2]
        self.display_scale = min(self.canvas_width / orig_w, self.canvas_height / orig_h)
        self.display_width = int(orig_w * self.display_scale)
        self.display_height = int(orig_h * self.display_scale)
        self.offset_x = (self.canvas_width - self.display_width) // 2
        self.offset_y = (self.canvas_height - self.display_height) // 2

        resized_img = cv2.resize(self.original_cv_img, (self.display_width, self.display_height))
        cv_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img_rgb)
        self.display_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW,
                                 image=self.display_img, state='disabled')

        self.detect_btn.config(state=tk.NORMAL)
        self.modify_btn.config(state=tk.DISABLED)
        self.confirm_btn.config(state=tk.DISABLED)
        self.measure_btn.config(state=tk.DISABLED)
        self.mode_label.pack_forget()
        self.draggable_points = []
        self.octagon_id = None
        self.octagon_confirmed = False

    def detect(self):
        if self.image_path is None:
            messagebox.showwarning("警告", "まず画像をアップロードしてください。")
            return
        try:
            out_img, self.mask_final = detect_bedsores(self.image_path)
            self.detected_img = out_img.copy()
        except Exception as e:
            messagebox.showerror("エラー", f"検出処理でエラーが発生しました:\n{e}")
            return

        resized_detected = cv2.resize(self.detected_img, (self.display_width, self.display_height))
        cv_img_rgb = cv2.cvtColor(resized_detected, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img_rgb)
        self.display_img = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW,
                                 image=self.display_img, state='disabled')

        self.modify_btn.config(state=tk.NORMAL)
        self.confirm_btn.config(state=tk.DISABLED)
        self.measure_btn.config(state=tk.DISABLED)
        self.mode_label.pack_forget()
        self.draggable_points = []
        self.octagon_id = None
        self.octagon_confirmed = False

    def modify(self):
        self.mode_label.pack(side=tk.RIGHT, padx=5)
        # もし以前「確定」している八角形があれば、その座標から編集状態にする
        if self.octagon_confirmed and self.octagon_id is not None:
            coords = self.canvas.coords(self.octagon_id)
            if len(coords) < 16:
                messagebox.showerror("エラー", "八角形の情報が不正です。")
                return
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        else:
            if self.mask_final is None:
                messagebox.showwarning("警告", "検出結果がありません。")
                return
            contours, _ = cv2.findContours(self.mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                messagebox.showerror("エラー", "有効な領域が検出されませんでした。")
                return
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour = largest_contour.astype(np.float32)
            largest_contour = largest_contour * self.display_scale
            largest_contour += np.array([[self.offset_x, self.offset_y]])
            points = get_evenly_spaced_points(largest_contour, num_points=8)  # ★ここの値を変更することで、修正するための円の数が変わる。

        for pt in self.draggable_points:
            self.canvas.delete(pt.hit_id)
            self.canvas.delete(pt.vis_id)
        self.draggable_points = []
        if self.octagon_id is not None:
            self.canvas.delete(self.octagon_id)
            self.octagon_id = None

        for (x, y) in points:
            pt = DraggablePoint(self.canvas, x, y, vis_radius=5, hit_radius=20, fill="blue",
                                callback=self.update_octagon)
            self.draggable_points.append(pt)
        coords = []
        for pt in self.draggable_points:
            coords.extend([pt.x, pt.y])
        self.octagon_id = self.canvas.create_polygon(*coords, outline="lime", fill="", width=2, tags="octagon")
        self.canvas.tag_raise(self.octagon_id)
        for pt in self.draggable_points:
            self.canvas.tag_raise(pt.vis_id)
            self.canvas.tag_raise(pt.hit_id)

        self.octagon_confirmed = False
        self.confirm_btn.config(state=tk.NORMAL)
        self.measure_btn.config(state=tk.DISABLED)

    def update_octagon(self):
        if self.octagon_id is None:
            return
        coords = []
        for pt in self.draggable_points:
            coords.extend([pt.x, pt.y])
        self.canvas.coords(self.octagon_id, *coords)

    def confirm(self):
        if not self.draggable_points:
            messagebox.showwarning("警告", "頂点が設定されていません。")
            return
        for pt in self.draggable_points:
            pt.disable()
        self.octagon_confirmed = True
        self.confirm_btn.config(state=tk.DISABLED)
        self.mode_label.pack_forget()
        # 「確定」後、計測ボタンを有効化
        self.measure_btn.config(state=tk.NORMAL)
        messagebox.showinfo("確定", "八角形の修正を確定しました。")

    def measure(self):
        # 「確定」済みでなければエラー
        if not self.octagon_confirmed:
            messagebox.showwarning("警告", "まず八角形を確定してください。")
            return

        # カメラと物体の距離をユーザーに入力してもらう（単位はmmとする例）
        distance = simpledialog.askfloat("距離入力", "カメラと物体との距離を入力してください (mm):")
        if distance is None or distance <= 0:
            messagebox.showerror("エラー", "有効な距離が入力されませんでした。")
            return

        # --- ここから計測処理 ---
        # まず、キャンバス上の八角形の頂点（表示座標）を元画像のピクセル座標に変換する
        # 変換方法： (canvas座標 - offset) / display_scale
        octagon_points = []
        for pt in self.draggable_points:
            orig_x = (pt.x - self.offset_x) / self.display_scale
            orig_y = (pt.y - self.offset_y) / self.display_scale
            octagon_points.append((orig_x, orig_y))

        # ピクセル単位での面積（シューレースの公式）
        area_pixels = polygon_area(octagon_points)

        # ピクセル単位での八角形の軸平行な外接矩形（bounding box）を求める
        xs = [p[0] for p in octagon_points]
        ys = [p[1] for p in octagon_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        # 正方形で囲む場合、必要な1辺の長さは幅と高さのうち大きい方
        square_side_pixels = max(bbox_width, bbox_height)

        # --- ピンホールカメラモデルによる実寸換算 ---
        # 1ピクセルあたりのセンサー上の長さ (mm)
        orig_img_width = self.original_cv_img.shape[1]
        pixel_size = SENSOR_WIDTH / orig_img_width  # mm/px

        # 実物の長さ = (ピクセル長 * pixel_size) * (distance / FOCAL_LENGTH)
        conversion_factor = (pixel_size * distance) / FOCAL_LENGTH  # mm per pixel in the object plane

        real_area = area_pixels * (conversion_factor ** 2)  # mm^2
        real_square_side = square_side_pixels * conversion_factor  # mm

        # 結果を表示
        result_text = (f"八角形の面積: {real_area:.2f} mm²\n"
                       f"正方形で囲んだ場合の1辺の長さ: {real_square_side:.2f} mm")
        messagebox.showinfo("計測結果", result_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = BedsoresDetectionApp(root)
    root.mainloop()
