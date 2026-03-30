import json

with open('hello.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_code = '''import cv2
import numpy as np
import time
import pygame
from tensorflow.keras.models import load_model

# Initialize pygame mixer for alarm sound
pygame.mixer.init()
alarm_sound = r"C:\\Users\\dy229\\OneDrive\\Desktop\\AI and ML\\Machine learning\\full project\\Driver_fatique_detection\\arpit_bala_alarm.mp3"

# Load your trained model
model = load_model(r"C:\\Users\\dy229\\OneDrive\\Desktop\\AI and ML\\Machine learning\\full project\\Driver_fatique_detection\\driver_fatigue_model.h5")

# Load Haar cascades for face & eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img_size = 224
CLOSED_THRESHOLD = 2  # seconds

eyes_closed_start = None
alarm_on = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_open = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 4)

        for (ex, ey, ew, eh) in eyes:
            eye_img = face_roi_color[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (img_size, img_size))
            eye_img = eye_img.astype("float32") / 255.0
            eye_img = np.expand_dims(eye_img, axis=0)  # (1, 224, 224, 3)

            prediction = model.predict(eye_img, verbose=0)[0][0]

            # prediction > 0.5 = open, <= 0.5 = closed
            if prediction > 0.5:
                label = "Open"
                color = (0, 255, 0)
                eyes_open = True
            else:
                label = "Closed"
                color = (0, 0, 255)

            cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            cv2.putText(frame, label, (x+ex, y+ey-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Track how long eyes have been closed
    if not eyes_open:
        if eyes_closed_start is None:
            eyes_closed_start = time.time()
        else:
            closed_duration = time.time() - eyes_closed_start
            cv2.putText(frame, f"Eyes Closed: {closed_duration:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if closed_duration >= CLOSED_THRESHOLD:
                cv2.putText(frame, "WAKE UP!!!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                # Play custom alarm sound
                if not alarm_on:
                    pygame.mixer.music.load(alarm_sound)
                    pygame.mixer.music.play(-1)  # Loop until eyes open
                    alarm_on = True
    else:
        eyes_closed_start = None
        if alarm_on:
            pygame.mixer.music.stop()
            alarm_on = False

    cv2.imshow("Driver Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
'''

# Convert the code string into notebook source format (list of lines)
lines = new_code.strip().split('\n')
source_lines = [line + '\n' for line in lines[:-1]]
source_lines.append(lines[-1])  # last line without trailing newline

nb['cells'][0]['source'] = source_lines
nb['cells'][0]['outputs'] = []
nb['cells'][0]['execution_count'] = None

with open('hello.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done! Notebook updated successfully.')
