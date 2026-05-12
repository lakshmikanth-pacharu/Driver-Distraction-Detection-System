import cv2
import mediapipe as mp
import os

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Image path
image_path = "converted_image.jpg"   # your RGB converted image
image = cv2.imread(image_path)

# Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Output folder (same folder)
output_folder = os.path.dirname(image_path)

# Find next available image number
count = 1
while os.path.exists(os.path.join(output_folder, f"annotated_{count}.jpg")):
    count += 1

# MediaPipe processing
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True) as face_mesh, \
        mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2) as hands:

    # Face detection
    face_results = face_mesh.process(rgb_image)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                None,
                mp_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                None,
                mp_styles.get_default_face_mesh_contours_style()
            )

    # Hand detection
    hand_results = hands.process(rgb_image)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

# Save output image
output_path = os.path.join(output_folder, f"annotated_{count}.jpg")
cv2.imwrite(output_path, image)

print(f"Image saved as {output_path}")