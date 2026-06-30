from flask import Flask, render_template, request, Response, jsonify
from ultralytics import YOLO
import cv2
import os
import time

app = Flask(__name__)

model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

signal_status = "RED"
ambulance_last_seen = 0
ambulance_count = 0
ambulance_present = False
video_path = None
video_finished = False

# realistic signal timings
GREEN_HOLD_TIME = 8
YELLOW_TO_GREEN = 2
GREEN_TO_RED_YELLOW = 3


def generate_frames(video_path):

    global signal_status
    global ambulance_last_seen
    global ambulance_count
    global ambulance_present
    global video_finished

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps if fps > 0 else 0.03

    frame_count = 0

    while True:
        start_time = time.time()

        success, frame = cap.read()

        # VIDEO FINISHED
        if not success:
            signal_status = "RED"
            video_finished = True
            break

        frame_count += 1
        ambulance_detected = False

        small_frame = cv2.resize(frame, (416, 416))

        # detect every 10 frames (faster playback)
        if frame_count % 10 == 0:
            results = model.predict(
                small_frame, conf=0.7, imgsz=416, device="cpu", verbose=False
            )

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

            if ambulance_detected:
                ambulance_last_seen = time.time()

                if not ambulance_present:
                    ambulance_count += 1
                    ambulance_present = True

                if signal_status == "RED":
                    signal_status = "YELLOW"
                    time.sleep(YELLOW_TO_GREEN)

                    signal_status = "GREEN"

                x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                scale_x = frame.shape[1] / 416
                scale_y = frame.shape[0] / 416

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if (
            signal_status == "GREEN"
            and time.time() - ambulance_last_seen > GREEN_HOLD_TIME
        ):
            ambulance_present = False

            signal_status = "YELLOW"
            time.sleep(GREEN_TO_RED_YELLOW)

            signal_status = "RED"

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        elapsed = time.time() - start_time
        if frame_delay > elapsed:
            time.sleep(frame_delay - elapsed)

    cap.release()


@app.route("/", methods=["GET", "POST"])
def index():

    global video_path
    global ambulance_count
    global signal_status
    global ambulance_present
    global video_finished

    if request.method == "POST":
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))

        file = request.files["video"]

        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        ambulance_count = 0
        signal_status = "RED"
        ambulance_present = False
        video_finished = False

        return render_template("index.html", video=True)

    return render_template("index.html", video=False)


@app.route("/video_feed")
def video_feed():

    global video_path

    if not video_path:
        return "No video uploaded"

    return Response(
        generate_frames(video_path),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/signal_status")
def signal():

    global signal_status
    global ambulance_count
    global video_finished

    return jsonify(
        {"signal": signal_status, "count": ambulance_count, "finished": video_finished}
    )


if __name__ == "__main__":
    app.run(debug=True)
