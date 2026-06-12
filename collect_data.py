import cv2
import mediapipe as mp
import csv
import os

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# ---------------- DATASET ----------------
os.makedirs("dataset", exist_ok=True)
file_path = "dataset/gestures.csv"

# Create CSV header
if not os.path.exists(file_path):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

print("Saving to:", os.path.abspath(file_path))
print("Controls: D=draw | E=erase | N=none | ESC=exit")

# ---------------- STATE ----------------
frame_count = 0
current_label = None
save_flag = False

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    # -------- INPUT --------
    if key == ord('d'):
        current_label = "draw"
        print("Switched to DRAW")
    elif key == ord('e'):
        current_label = "erase"
        print("Switched to ERASE")
    elif key == ord('n'):
        current_label = "none"
        print("Switched to NONE")
    elif key == 27:
        break

    frame_count += 1
    save_flag = False

    # -------- HAND PROCESS --------
    if result.multi_hand_landmarks:
        print("Hand detected")

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Normalize
            wrist = hand.landmark[0]
            data = []
            for lm in hand.landmark:
                data.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            # -------- SAVE LOGIC (RELAXED) --------
            if current_label and len(data) == 63:
                if frame_count % 3 == 0:   # less strict than 5

                    print(f">>> SAVING {current_label} <<<")

                    with open(file_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(data + [current_label])

                    save_flag = True

    else:
        print("No hand detected")

    # -------- UI --------
    cv2.putText(frame, f"Label: {current_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if current_label:
        cv2.putText(frame, f"Recording: {current_label}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if save_flag:
        cv2.putText(frame, "Saved!", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Collect Data", frame)

cap.release()
cv2.destroyAllWindows()