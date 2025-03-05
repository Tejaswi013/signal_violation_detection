import cv2
from ultralytics import YOLO

# Load YOLOv8 model (Nano version for fast inference)
model = YOLO("yolov8n.pt")

# Set the video file path
video_path = "clip.mp4"  # Change this to your actual video filename
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends

    # Perform YOLOv8 object detection
    results = model(frame)

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID
            label = model.names[class_id]  # Object class name

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("YOLO Detection", frame)

    # Write the frame to output video
    out.write(frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
