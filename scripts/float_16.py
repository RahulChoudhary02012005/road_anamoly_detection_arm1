import cv2
import numpy as np
import time
import os
import tflite_runtime.interpreter as tflite

# ==============================
# PATHS
# ==============================
MODEL_PATH = "/home/raspberrypi/Documents/MODELs/F1/weights/best_saved_model/best_float16.tflite"
VIDEO_PATH = "/home/raspberrypi/Documents/ARM_Input/15fps_input1.mp4"
OUTPUT_PATH = "/home/raspberrypi/Documents/MODELs/F1/Output.1/ultra_fast11.mp4"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ==============================
# SETTINGS
# ==============================
CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.45
NUM_THREADS = 4
FRAME_SKIP = 2   # Detection every 2 frames

CLASS_NAMES = [
    "HMV",
    "LMV",
    "Pedestrian",
    "RoadDamages",
    "SpeedBump",
    "UnsurfacedRoad"
]

# ==============================
# LOAD MODEL
# ==============================
interpreter = tflite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=NUM_THREADS
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

print("Model expects:", input_h, "x", input_w)

# ==============================
# LOAD VIDEO
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

fps_input = cap.get(cv2.CAP_PROP_FPS)

# Lower output resolution (big FPS gain)
PROCESS_W = 854
PROCESS_H = 480

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps_input,
    (PROCESS_W, PROCESS_H)
)

start_time = time.time()
frame_count = 0

last_boxes = []
last_ids = []
last_scores = []

# ==============================
# DETECTION LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize only for output
    frame_small = cv2.resize(frame, (PROCESS_W, PROCESS_H))

    # -------------------------
    # RUN INFERENCE (FRAME SKIP)
    # -------------------------
    if frame_count % FRAME_SKIP == 0:

        img = cv2.resize(frame, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        pred = interpreter.get_tensor(output_details[0]['index'])[0]

        # Some exports need transpose
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_raw = pred[:, :4]
        scores = pred[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        mask = confidences > CONF_THRESHOLD

        boxes_raw = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        boxes = []

        for i in range(len(boxes_raw)):
            x, y, bw, bh = boxes_raw[i]

            x1 = int((x - bw/2) * PROCESS_W)
            y1 = int((y - bh/2) * PROCESS_H)
            w_box = int(bw * PROCESS_W)
            h_box = int(bh * PROCESS_H)

            boxes.append([x1, y1, w_box, h_box])

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences.tolist(),
            CONF_THRESHOLD,
            NMS_THRESHOLD
        )

        last_boxes = []
        last_ids = []
        last_scores = []

        if len(indices) > 0:
            for i in indices.flatten():
                last_boxes.append(boxes[i])
                last_ids.append(class_ids[i])
                last_scores.append(confidences[i])

    # -------------------------
    # DRAW LAST DETECTIONS
    # -------------------------
    for i in range(len(last_boxes)):
        x, y, bw, bh = last_boxes[i]
        cid = int(last_ids[i])

        label = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
        conf = last_scores[i]

        cv2.rectangle(frame_small,
                      (x, y),
                      (x + bw, y + bh),
                      (0, 255, 0), 2)

        cv2.putText(frame_small,
                    f"{label} {conf:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0), 2)

    fps = frame_count / (time.time() - start_time)

    cv2.putText(frame_small,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255), 2)

    out.write(frame_small)

    # ⚠ Display disabled for higher FPS
    # cv2.imshow("Detection", frame_small)
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break

cap.release()
out.release()

print("Saved at:", OUTPUT_PATH)
print("Final FPS:", fps)
