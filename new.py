import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime

# Prompt user for the object to detect
target_object = input("Enter the object you want to detect (e.g., person, car, dog): ").strip().lower()

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 nano model

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform object detection
        results = model(frame)

        # Process results
        target_detected = False
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            names = result.names  # Class names

            # Draw bounding boxes
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{names[cls_id]} {conf:.2f}"
                    if names[cls_id].lower() == target_object:
                        # Draw red rectangle for target object
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        target_detected = True
                    else:
                        # Draw green rectangle for other objects
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put label above the rectangle
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if names[cls_id].lower() == target_object else (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Save image and exit if target object is detected
        if target_detected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detected_{target_object}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            break

        # Check for 'q' key or window close event
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
