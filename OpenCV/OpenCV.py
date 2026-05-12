import cv2
import os

cap = cv2.VideoCapture(0)

print("Press 's' to capture and save RGB image")
print("Press 'q' to quit")

# Step 1: Find existing images and set counter
existing_images = [f for f in os.listdir() if f.startswith("webcam_rgb_") and f.endswith(".jpg")]

if existing_images:
    image_count = len(existing_images) + 1
else:
    image_count = 1

while True:
    ret, frame = cap.read()   # BGR frame

    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Webcam Feed (BGR)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        filename = f"webcam_rgb_{image_count}.jpg"
        cv2.imwrite(filename, rgb_image)

        print(f"Saved {filename}")
        image_count += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
