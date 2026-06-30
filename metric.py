from ultralytics import YOLO
import time

# Load trained model
model = YOLO("best.pt")

# -----------------------------
# 1. Run dataset evaluation
# -----------------------------
metrics = model.val(data="data/data.yaml", split="test")

precision = metrics.box.p
recall = metrics.box.r
map50 = metrics.box.map50
map5095 = metrics.box.map

# -----------------------------
# 2. Compute F1 Score
# -----------------------------
f1_score = 2 * (precision * recall) / (precision + recall)

# -----------------------------
# 3. Measure Inference Time
# -----------------------------
start = time.time()

model.predict("data/images/test", save=False)

end = time.time()

inference_time = end - start

# -----------------------------
# 4. Compute FPS
# -----------------------------
fps = 1 / inference_time

# -----------------------------
# 5. Print Results
# -----------------------------
print("\nModel Evaluation Metrics\n")

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("mAP@0.5:", map50)
print("mAP@0.5:0.95:", map5095)
print("Inference Time (seconds):", inference_time)
print("FPS:", fps)
