import cv2

# Set the correct video path
video_path = r"C:\Users\tejas\Desktop\traffic\clip.mp4"  # Replace with your actual filename

# Load the video
cap = cv2.VideoCapture(video_path)

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read and display frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when the video ends

    # Show the video
    cv2.imshow("Traffic Video", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
