import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define video path
video_path = "clip_new.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("violation_output_new.mp4", fourcc, fps, (width, height))

# Global variable for stop line
STOP_LINE_Y = None
violating_vehicles = {}  # Dictionary to store violation timestamps

# Mouse click callback function to select stop line
def set_stop_line(event, x, y, flags, param):
    global STOP_LINE_Y
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click
        STOP_LINE_Y = y
        print(f"Stop line set at Y = {STOP_LINE_Y}")

# Show first frame to set stop line
ret, first_frame = cap.read()
if ret:
    cv2.imshow("Set Stop Line", first_frame)
    cv2.setMouseCallback("Set Stop Line", set_stop_line)
    print("Click on the video to set the stop line...")

    # Wait until user sets stop line
    while STOP_LINE_Y is None:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()

    cv2.destroyAllWindows()  # Close window after setting stop line

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()  # Get current timestamp

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = model.names[class_id]

            # Detect vehicles
            if label in ["car", "truck", "motorcycle"]:
                y_center = (y1 + y2) // 2
                vehicle_id = (x1, y1, x2, y2)  # Using bounding box coordinates to track vehicles

                # 🚨 NEW RULE: If the vehicle moves ABOVE the stop line, record violation time
                if y_center < STOP_LINE_Y and vehicle_id not in violating_vehicles:
                    violating_vehicles[vehicle_id] = current_time  # Store violation timestamp

                # Check if violation time is still within 1 second
                if vehicle_id in violating_vehicles and (current_time - violating_vehicles[vehicle_id]) < 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "VIOLATION!", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Reset to green after 1 sec

            # Display label
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw stop line
    cv2.line(frame, (0, STOP_LINE_Y), (width, STOP_LINE_Y), (0, 0, 255), 2)

    # Show video
    cv2.imshow("Violation Detection", frame)
    out.write(frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
