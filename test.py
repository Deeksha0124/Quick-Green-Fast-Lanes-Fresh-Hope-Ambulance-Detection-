from ultralytics import YOLO
import os
import cv2

# Load trained model
model = YOLO("best.pt")

input_folder = "test_images"
output_folder = "output_images"

os.makedirs(output_folder, exist_ok=True)

print("\n🚦 Starting Single Ambulance Detection...\n")

for image_name in os.listdir(input_folder):

    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        continue

    print(f"🔍 Processing: {image_name}")

    results = model.predict(
        source=image,
        conf=0.75,   # high confidence
        save=False
    )

    best_box = None
    best_conf = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])

            # Only ambulance class
            if label.lower() == "ambulance":

                # Keep only highest confidence
                if confidence > best_conf:
                    best_conf = confidence
                    best_box = box

    if best_box is not None:

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            image,
            f"Ambulance {best_conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        print("🚑 Ambulance Detected")

    else:
        print("❌ No Ambulance Detected")

    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)

print("\n✅ Detection Completed. Check 'output_images' folder.")