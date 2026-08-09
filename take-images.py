import cv2
import os
import time

# Configuration
NUM_IMAGES = 200
INTERVAL = 2  # seconds
OUTPUT_FOLDER = "tiger"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press 'q' to quit early.")

last_capture = time.time() - INTERVAL
image_count = 0

while image_count < NUM_IMAGES:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    current_time = time.time()
    remaining = max(0, INTERVAL - (current_time - last_capture))

    # Display info
    cv2.putText(
        frame,
        f"Captured: {image_count}/{NUM_IMAGES}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Next capture: {remaining:.1f}s",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Webcam Capture", frame)

    # Capture every INTERVAL seconds
    if current_time - last_capture >= INTERVAL:
        image_count += 1
        filename = os.path.join(
            OUTPUT_FOLDER,
            f"image_{image_count:03d}.jpg"
        )
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        last_capture = current_time

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()

print("Done.")