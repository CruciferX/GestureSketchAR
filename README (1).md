# ✋ AirCanvas

A real-time hand gesture drawing app using computer vision. Point your finger to draw, raise your palm to erase — no mouse or touch needed, just your hand and a camera.

---

## 🎯 Features

- **Draw in the air** — lift your index finger to start drawing on screen
- **Palm eraser** — raise your open palm to erase
- **Real-time tracking** — smooth, low-latency hand detection via webcam
- **No hardware needed** — works with any standard camera

---

## 🖥️ Demo

> Point finger → draws on canvas  
> Open palm → erases  
> No finger up → nothing happens (idle mode)

---

## 🛠️ Tech Stack

- **Python** — core language
- **OpenCV** — camera feed and canvas rendering
- **MediaPipe** — hand landmark detection

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed. Then install the required libraries:

```bash
pip install opencv-python mediapipe numpy
```

### Run the app

```bash
python aircanvas.py
```

Your webcam will open automatically. Show your hand to the camera and start drawing!

---

## 🤚 Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Draw |
| 🖐️ Open palm | Erase |
| ✊ Fist / no gesture | Idle (nothing happens) |

---

## 📁 Project Structure

```
air-canvas/
├── aircanvas.py       # Main application
├── README.md          # You are here
└── requirements.txt   # Dependencies
```

---

## 🙌 Contributing

Pull requests are welcome! If you have ideas for new gestures or features, feel free to open an issue first to discuss.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

Made with ❤️ and a wave of the hand.
