import cv2
import os
import random
from ultralytics import YOLO
from tracker import Tracker

video_path = os.path.join('.','data','test1_cut.mp4')
result_path = os.path.join('.','result.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
classes = ["Civilian", "Soldier"]
cap_out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
model = YOLO('surveillance.pt')
tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0,255), random.randint(0, 255)) for j in range(15)]

detection_threshold = 0.5
while ret:
    results = model(frame)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score, class_id])
    tracker.update(frame, detections)

    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id
        class_id = track.class_id
        label = f"{classes[class_id]}: {track_id}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 2)
        cv2.putText(frame, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (colors[track_id % len(colors)]), 2, cv2.LINE_AA)

    cv2.imshow("Display", frame)
    cap_out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows
