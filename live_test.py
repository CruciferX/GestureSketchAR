import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

buffer = deque(maxlen=5)

def get_stable_prediction():
    if len(buffer) < 5:
        return "none"
    if all(p == buffer[0] for p in buffer):
        return buffer[0]
    return "none"

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "none"

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            wrist = hand.landmark[0]
            data = []
            for lm in hand.landmark:
                data.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            probs = model.predict_proba([data])[0]
            max_prob = max(probs)

            if max_prob > 0.7:
                pred = model.classes_[np.argmax(probs)]
            else:
                pred = "none"

            buffer.append(pred)
            gesture = get_stable_prediction()

    cv2.putText(frame, f"{gesture}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()