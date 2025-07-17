import cv2
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from datetime import datetime
from PIL import Image
import os
import tempfile
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Get user input for the target object
target_object = input("Enter the object you want to find: ").strip().lower()

# Initialize YOLOv8 for detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model = YOLO("yolov8n.pt").to(device)  # Use lighter YOLOv8n model

# Initialize Qwen-VL for image description
try:
    qwenvl_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        trust_remote_code=True,
        use_safetensors=True
    ).to(device).eval()  # Set model to evaluation mode
    qwenvl_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        trust_remote_code=True
    )
except Exception as e:
    logger.error(f"Failed to load Qwen-VL model or tokenizer: {e}")
    exit(1)

# Cache for object descriptions
description_cache = {}

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Could not open camera.")
    exit(1)

# Ensure OpenCV window
cv2.namedWindow("Real-Time Detection with Qwen-VL", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Real-Time Detection with Qwen-VL", cv2.WND_PROP_VISIBLE, 1)

frame_count = 0
max_processing_time = 2.0  # Maximum time (seconds) for processing per frame
last_processed_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read frame.")
            break

        # Preprocess frame
        frame_resized = cv2.resize(frame, (320, 320))  # Even smaller resolution
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Display raw frame if processing is too slow
        current_time = time.time()
        if current_time - last_processed_time > max_processing_time:
            cv2.imshow("Real-Time Detection with Qwen-VL", cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))
            logger.warning("Processing too slow, showing raw frame.")
        else:
            # Run YOLO every 10 frames
            if frame_count % 10 == 0:
                start_time = time.time()
                results = model.predict(frame_rgb, conf=0.5, device=device, verbose=False)
                boxes = results[0].boxes.xyxy
                scores = results[0].boxes.conf
                target_detected = False

                # Process only if boxes are detected
                if len(boxes) > 0:
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame_rgb[y1:y2, x1:x2]
                        if crop.size == 0:
                            logger.warning("Empty crop detected, skipping.")
                            continue

                        # Save crop to temporary file
                        crop_pil = Image.fromarray(crop)
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                            temp_image_path = temp_file.name
                            crop_pil.save(temp_image_path)

                        # Check cache or process with Qwen-VL only if not in cache
                        box_key = f"{x1}_{y1}_{x2}_{y2}_{score:.2f}"
                        if box_key not in description_cache:
                            try:
                                query = "Describe this object in one sentence."
                                inputs = qwenvl_tokenizer.from_list_format([
                                    {'image': temp_image_path},
                                    {'text': query}
                                ])
                                inputs = qwenvl_tokenizer(inputs, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    outputs = qwenvl_model.generate(**inputs, max_new_tokens=10)
                                description = qwenvl_tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
                                description_cache[box_key] = description
                            except Exception as e:
                                logger.error(f"Qwen-VL processing failed: {e}")
                                description = "Unknown"
                            finally:
                                os.remove(temp_image_path)  # Clean up
                        else:
                            description = description_cache[box_key]

                        # Visualize detection
                        box_color = (0, 0, 255) if target_object in description else (0, 255, 0)
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), box_color, 2)
                        label = f"{description.split('.')[0]}: {score:.2f}"
                        cv2.putText(frame_resized, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                        # Check for target object
                        if target_object in description:
                            target_detected = True
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"detected_{target_object}_at_{timestamp}.jpg"
                            cv2.imwrite(filename, cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))
                            logger.info(f"Image with {target_object} saved as {filename}")

                last_processed_time = time.time()
                logger.info(f"Processing time: {last_processed_time - start_time:.2f} seconds")

            # Display processed frame
            cv2.imshow("Real-Time Detection with Qwen-VL", cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))

        # Check for key press
        key = cv2.waitKey(10) & 0xFF  # Increased delay to handle lag
        if key == ord("q") or key == ord("Q"):
            logger.info("Exiting on 'q' key")
            break
        elif key in [13, 32]:  # Capture photo on 'Enter' or 'Space'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manual_capture_{timestamp}.jpg"
            cv2.imwrite(filename, cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))
            logger.info(f"Image saved as {filename}")

        frame_count += 1

except KeyboardInterrupt:
    logger.info("Program terminated by user (Ctrl+C or similar).")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
