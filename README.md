# 🚗 Driver Fatigue Detection System

This project is a real-time driver fatigue detection system built with Python, OpenCV, and Deep Learning (TensorFlow/Keras). It continuously monitors a driver's eyes using a webcam and triggers a high-frequency beep alarm if the eyes remain closed for more than a specified threshold (e.g., 1 second), helping to prevent accidents caused by drowsiness.

## ✨ Features
- **Real-Time Tracking**: Uses OpenCV Haar cascades to instantly detect faces and isolate eye regions from a live webcam feed.
- **Deep Learning Inference**: Passes the isolated eye patches through a trained neural network (`driver_fatigue_model.h5`) to predict whether the eyes are "Open" or "Closed".
- **Dynamic Warning System**: Calculates the exact duration of eye closure. If it exceeds the safety threshold, it fires a `WAKE UP!!!` visual warning overlaid on the screen and triggers an audio alarm.
- **Optimized for Windows**: Uses `winsound` for built-in system beeps and properly hooks into DirectShow (`cv2.CAP_DSHOW`) for smooth webcam integration.

## 🛠️ Prerequisites & Setup

You will need the following Python libraries installed. You can install them via pip:

```bash
pip install opencv-python numpy tensorflow pygame
```
> **Note**: While the repository might contain legacy files for a `pygame` audio alarm, the current `hello.ipynb` logic uses `winsound` (built-in for Windows) so `pygame` is optional.

You also need a trained model named `driver_fatigue_model.h5` placed in the project directory. *(Note: This file and the training datasets are deliberately ignored in `.gitignore` due to file size constraints).*

## 🚀 Usage

To run the fatigue detection logic locally, you can use the Jupyter Notebook implementation:
1. Open the project folder in your preferred IDE (e.g., VS Code or Jupyter).
2. Open `hello.ipynb`.
3. Restart your kernel if you've run the camera previously (to ensure it isn't locked).
4. Run the main processing cell. 
5. The webcam will open, draw boundary boxes around your face and eyes, and monitor your eye state.
6. **To quit**: Press the `q` key with the video window active.

## 📂 Project Structure

- `hello.ipynb`: The main real-time webcam inference and alarm loop.
- `Driver_Fatigue_Detection_Colab.ipynb`: Notebook containing the original deep learning model training routines designed for Google Colab.
- `organize_eyes.py`: A utility script used to iterate through the raw image dataset and properly categorize images into `open_eyes` and `closed_eyes` folders for training.
- `update_notebook.py` / `update_notebook_beep.py`: Automated scripts previously used to clean and rewrite notebook JSON sources to swap alarm systems.
- `.gitignore`: Security configuration to avoid uploading oversized `.h5` model files, datasets, and local environment variables.

## 🤝 Acknowledgements
- Haar Cascades by [OpenCV](https://opencv.org/).
- Deep Learning framework by [TensorFlow / Keras](https://www.tensorflow.org/).
