import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque, Counter

# ---------------- LOAD MODEL ----------------
model = joblib.load("gesture_model.pkl")

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# Force HD (optional but good)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Camera error")
    exit()

frame = cv2.flip(frame, 1)
h, w, _ = frame.shape

# Canvas same size
canvas = np.zeros((h, w, 3), dtype=np.uint8)

cv2.namedWindow("Gesture Drawing System", cv2.WINDOW_NORMAL)

# ---------------- SMOOTHING ----------------
buffer = deque(maxlen=5)

def get_stable_prediction():
    if len(buffer) < 3:
        return "none"
    
    most_common, count = Counter(buffer).most_common(1)[0]
    
    if count >= 2:
        return most_common
    
    return "none"

# ---------------- CROP TO 16:10 ----------------
def crop_to_16_10(image):
    h, w = image.shape[:2]
    
    target_ratio = 16 / 10
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Crop sides
        new_w = int(h * target_ratio)
        x_start = (w - new_w) // 2
        cropped = image[:, x_start:x_start+new_w]
    else:
        # Crop top/bottom (rare)
        new_h = int(w / target_ratio)
        y_start = (h - new_h) // 2
        cropped = image[y_start:y_start+new_h, :]

    return cropped

# ---------------- MAIN LOOP ----------------
prev_point = None

print("Gesture Drawing Started | ESC to exit | Press C to clear")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    pred = "none"
    gesture = "none"

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            # Normalize landmarks
            wrist = hand.landmark[0]
            data = []
            for lm in hand.landmark:
                data.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            # Prediction
            probs = model.predict_proba([data])[0]
            max_prob = max(probs)

            if max_prob > 0.5:
                pred = model.classes_[np.argmax(probs)]
            else:
                pred = "none"

            buffer.append(pred)
            gesture = get_stable_prediction()

            # Finger tip
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # DRAW
            if gesture == "draw":
                if prev_point is None:
                    prev_point = (x, y)

                cv2.line(canvas, prev_point, (x, y), (255, 255, 255), 3)
                prev_point = (x, y)

            # ERASE
            elif gesture == "erase":
                cv2.circle(canvas, (x, y), 40, (0, 0, 0), -1)
                prev_point = None

            else:
                prev_point = None

    # Merge
    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Debug
    cv2.putText(output, f"Pred: {pred}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(output, f"Gesture: {gesture}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # -------- FULLSCREEN FIX --------
    cropped = crop_to_16_10(output)
    display = cv2.resize(cropped, (1280, 800))  # 16:10

    cv2.imshow("Gesture Drawing System", display)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('c'):
        canvas[:] = 0

cap.release()
cv2.destroyAllWindows()