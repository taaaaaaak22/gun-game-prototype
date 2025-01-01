import cv2
from ultralytics import YOLO
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import websocket
import json
import threading

# YOLOモデルのロード
yolo_model = YOLO("yolov8n.pt")

# YOLOのログを非表示にする
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# MediaPipe Poseのセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 日本語フォントのパスを指定
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf" 

# スコアの初期化
score = 0
last_target_part = ""

# ライフの初期化
life = 5
game_over = False
damage_flash = False
flash_counter = 0

# 頭に該当するランドマークインデックスと部位名
HEAD_PARTS = {
    0: "鼻", 1: "左目内側", 2: "左目", 3: "左目外側", 
    4: "右目内側", 5: "右目", 6: "右目外側", 
    7: "左耳", 8: "右耳"
}

# プレイヤーIDを定義
player_id = "A"

# カメラを開く
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("カメラを開けませんでした")
    exit()

def on_message(ws, message):
    global life, game_over, damage_flash
    data = json.loads(message)
    print(f"受信メッセージ: {data}")  # デバッグ用にメッセージ内容を出力
    if data['type'] == 'damage' and data.get('targetId') == player_id:
        life -= data['damage']
        damage_flash = True
        print(f"ダメージを受けました: {data['damage']} (残りライフ: {life})")
        if life <= 0:
            game_over = True

def on_open(ws):
    ws.send(json.dumps({"type": "register", "playerId": player_id}))

def on_error(ws, error):
    print(f"WebSocketエラー: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket接続が閉じられました")

# WebSocketサーバーに接続
ws = websocket.WebSocketApp(
    "ws://localhost:8080",
    on_message=on_message,
    on_open=on_open,
    on_error=on_error,
    on_close=on_close
)

# WebSocketのスレッドを開始
ws_thread = threading.Thread(target=ws.run_forever)
ws_thread.start()

# WebSocketの接続が確立されるまで待機
while not ws.sock or not ws.sock.connected:
    pass

while True:
    # フレームを取得
    ret, frame = camera.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # フレームのコピーを作成
    annotated_frame = frame.copy()

    # ダメージを受けた際の赤い点滅
    if damage_flash:
        flash_counter += 1
        if flash_counter % 10 < 5:
            annotated_frame[:, :, 2] = np.maximum(annotated_frame[:, :, 2], 100)  # 赤色を強調
        if flash_counter > 20:
            damage_flash = False
            flash_counter = 0

    # フレームサイズを取得
    frame_height, frame_width = frame.shape[:2]

    # 中央エリアを定義 (幅と高さの20%の範囲)
    center_x1 = int(frame_width * 0.4)
    center_y1 = int(frame_height * 0.4)
    center_x2 = int(frame_width * 0.6)
    center_y2 = int(frame_height * 0.6)

    # 中央エリアの中心座標
    center_x = (center_x1 + center_x2) // 2
    center_y = (center_y1 + center_y2) // 2

    # 中央エリアを描画
    cv2.rectangle(annotated_frame, (center_x1, center_y1), (center_x2, center_y2), (0, 255, 0), 2)

    # YOLOで人物を検出
    results = yolo_model(frame, classes=[0])  # '0'はCOCOデータセットの人物クラスID

    closest_part = None
    min_distance = float('inf')
    target_part_name = "人物がいません"
    specific_head_part = ""  # 頭部の具体的な部位名
    person_in_center = False  # 中央エリアに人物がいるか

    for result in results[0].boxes:
        # バウンディングボックス座標を取得
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # バウンディングボックスが中央エリアに重なっているかを判定
        if not (x2 < center_x1 or x1 > center_x2 or y2 < center_y1 or y1 > center_y2):
            person_in_center = True

            # 検出された領域を切り取り
            person_roi = frame[y1:y2, x1:x2]

            # 姿勢推定を適用
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(person_rgb)

            if pose_results.pose_landmarks:
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    # ランドマークの座標を計算（相対座標を絶対座標に変換）
                    lx = int(x1 + landmark.x * (x2 - x1))
                    ly = int(y1 + landmark.y * (y2 - y1))

                    # 中央エリア内の部位のみ対象
                    if center_x1 <= lx <= center_x2 and center_y1 <= ly <= center_y2:
                        # 中央点との距離を計算
                        distance = ((center_x - lx) ** 2 + (center_y - ly) ** 2) ** 0.5

                        # 最も近い部位を更新
                        if distance < min_distance:
                            min_distance = distance
                            closest_part = (lx, ly)
                            target_part_name = "頭" if idx in HEAD_PARTS else "それ以外"
                            specific_head_part = HEAD_PARTS.get(idx, "")

    # スコア対象の部位を描画
    if closest_part:
        cv2.circle(annotated_frame, closest_part, 10, (0, 0, 255), -1)  # 赤い円で表示
        last_target_part = target_part_name

    # 日本語を描画するためにPillowを使用
    annotated_frame_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_frame_pil)
    font = ImageFont.truetype(FONT_PATH, 24)  # フォントサイズを指定

    # スコア表示
    draw.text((10, 10), f"スコア: {score}", font=font, fill=(255, 255, 255))

    # ライフ表示
    draw.text((10, 90), f"ライフ: {life}", font=font, fill=(255, 255, 255))

    # ゲームオーバー表示
    if game_over:
        draw.text((frame_width // 2 - 100, frame_height // 2), "ゲームオーバー", font=font, fill=(255, 0, 0))

    # 対象部位名の表示
    if person_in_center:
        if target_part_name == "頭" and specific_head_part:
            draw.text((10, 50), f"対象部位: {target_part_name} ({specific_head_part})", font=font, fill=(255, 255, 255))
        elif target_part_name == "それ以外":
            draw.text((10, 50), f"対象部位: {target_part_name}", font=font, fill=(255, 255, 255))
    else:
        draw.text((10, 50), "人物がいません", font=font, fill=(255, 0, 0))

    # Pillow画像をOpenCV形式に戻す
    annotated_frame = cv2.cvtColor(np.array(annotated_frame_pil), cv2.COLOR_RGB2BGR)

    # 映像を表示
    cv2.imshow("YOLO + Pose Estimation with Scoring", annotated_frame)

    # キー入力を処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q'キーで終了
        break
    elif key == ord('b') and person_in_center and not game_over:  # 'b'キーでスコアを加算
        if target_part_name == "頭":
            score += 2
            damage = 2
        elif target_part_name == "それ以外":
            score += 1
            damage = 1

        # ダメージをサーバーに送信
        target_id = "A"  # ダミーのターゲットID
        ws.send(json.dumps({"type": "damage", "fromId": player_id, "targetId": target_id, "damage": damage}))

    elif key == ord('d') and not game_over:  # 'd'キーでダメージを受ける（デバッグ用）
        life -= 1
        damage_flash = True
        if life <= 0:
            game_over = True

# リソースを解放
camera.release()
cv2.destroyAllWindows()
ws.close()
ws_thread.join()

