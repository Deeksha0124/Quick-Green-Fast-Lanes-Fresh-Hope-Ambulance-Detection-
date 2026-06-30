from ultralytics import YOLO
import cv2
import time

# ------------------------------
# Load trained model
# ------------------------------
model = YOLO("best.pt")

# ------------------------------
# Input & Output Video
# ------------------------------
input_video = "demo_video.mp4"   # put your video name here
output_video = "output_video.mp4"

cap = cv2.VideoCapture(input_video)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    output_video,
    fourcc,
    20.0,
    (int(cap.get(3)), int(cap.get(4)))
)

# ------------------------------
# Signal Logic Variables
# ------------------------------
ambulance_last_seen = 0
signal_status = "RED"

print("\n🚦 Starting Video Detection...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.75, save=False)

    ambulance_detected = False
    best_box = None
    best_conf = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])

            if label.lower() == "ambulance":
                if confidence > best_conf:
                    best_conf = confidence
                    best_box = box
                    ambulance_detected = True

    # If ambulance detected
    if ambulance_detected:
        ambulance_last_seen = time.time()
        signal_status = "GREEN"

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame,
                    f"Ambulance {best_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    # If no ambulance for 5 seconds → RED
    elif time.time() - ambulance_last_seen > 5:
        signal_status = "RED"

    # Draw Signal Text
    color = (0, 255, 0) if signal_status == "GREEN" else (0, 0, 255)

    cv2.putText(frame,
                f"SIGNAL: {signal_status}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3)

    # Write frame
    out.write(frame)

    # Show live window
    cv2.imshow("Ambulance Signal System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n✅ Video Processing Completed.")
print("Saved as:", output_video)