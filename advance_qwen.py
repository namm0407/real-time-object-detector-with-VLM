import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import sys
import win32gui
import win32con

# Prompt user for the object to detect
target_input = input("Enter the object you want to detect (e.g., person, car, dog, red cup): ").strip().lower()

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 nano model

# Get valid class names from the model
valid_classes = [name.lower() for name in model.names.values()]

# Define valid colors for filtering (basic colors for simplicity)
valid_colors = ['red', 'blue', 'green', 'yellow', 'white', 'black']

# Define HSV range for red (two ranges due to red wrapping around hue)
red_lower1 = np.array([0, 70, 50])    # Lower red range
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 70, 50])  # Upper red range
red_upper2 = np.array([180, 255, 255])

# Function to check if a region is predominantly red
def is_red_region(roi):
    # Convert ROI to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
    mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    # Calculate the percentage of red pixels
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    red_percentage = (red_pixels / total_pixels) * 100
    # Consider it red if at least 20% of pixels are red (adjustable threshold)
    return red_percentage > 20

# Preprocess target input to extract object and color
def extract_object_and_color(target):
    words = target.split()
    color = None
    object_name = None
    # Try to find a valid color and object
    for word in words:
        if word in valid_colors:
            color = word
        if word in valid_classes:
            object_name = word
    # Fallback: assume last word is the object
    if not object_name and words and words[-1] in valid_classes:
        object_name = words[-1]
    if not object_name:
        print(f"Warning: No valid object found in '{target}'. Valid classes: {', '.join(valid_classes)}")
        return None, None
    return object_name, color

# Extract object and color
target_object, target_color = extract_object_and_color(target_input)
if target_object is None:
    print("Error: Invalid object name. Exiting.")
    sys.exit(1)
print(f"Detecting: {target_object} {'with color ' + target_color if target_color else ''}")

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Set window name and ensure it’s created
window_name = 'Object Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Force focus on the window
def set_window_focus(window_name):
    try:
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                 win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            print(f"Focused on window: {window_name}")
        else:
            print(f"Window '{window_name}' not found.")
    except Exception as e:
        print(f"Error setting window focus: {e}")

set_window_focus(window_name)

# Object detection loop
image_path = None
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
                    is_target = names[cls_id].lower() == target_object
                    # Check color if specified (e.g., red)
                    if is_target and target_color == 'red':
                        # Extract ROI
                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0:  # Skip empty ROIs
                            continue
                        # Check if ROI is red
                        if not is_red_region(roi):
                            is_target = False
                    if is_target:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        target_detected = True
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                               (0, 0, 255) if is_target else (0, 255, 0), 2)

        # Display the frame
        cv2.imshow(window_name, frame)

        # Save image and exit if target object is detected
        if target_detected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"detected_{target_object}_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            break

        # Check for keypress
        key = cv2.waitKey(50)  # 50ms delay
        if key != -1:  # Only print if a key is pressed
            print(f"Key pressed: {key} (ASCII: {chr(key) if 32 <= key <= 126 else 'non-printable'})")
        if key == ord('q'):
            print("Exiting: 'q' key pressed.")
            break
        if key == 27:  # Esc key
            print("Exiting: Esc key pressed.")
            break

        # Check if window is closed
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Exiting: Window closed.")
                break
        except cv2.error:
            print("Exiting: Window closed or error in window property.")
            break

finally:
    # Release webcam resources
    cap.release()
    cv2.destroyAllWindows()
