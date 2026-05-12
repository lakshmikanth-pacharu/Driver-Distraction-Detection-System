#import Libraries
import cv2
import mediapipe as mp
import pygame
import time
import math
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.5)


# Initialize pygame for Buzzer
pygame.init()
pygame.mixer.init()
try:
    sound = pygame.mixer.Sound("buzzer.mp3")
except:
    print("Warning: Could not load buzzer.mp3 - audio alerts disabled")
    sound = None

# Constants (or) Threshold Values
CALIBRATION_TIME = 5
DISTRACTION_THRESHOLD = 3
YAWN_THRESHOLD = 0.35  # Mouth aspect ratio threshold
EAR_THRESHOLD = 0.21  # Adjusted for better sensitivity
EAR_CONSEC_FRAMES = 3  # Number of consecutive frames for blink detection
PERCLOS_THRESHOLD = 0.3
PHONE_DISTANCE_THRESHOLD = 0.05  # Relative to frame width
HAND_FACE_THRESHOLD = 0.25  # Relative to frame width
SEATBELT_HEIGHT_THRESHOLD = 0.8  # Shoulder Y position threshold

# State variables
calibrating = False
forward_head_position = None
closed_frames = 0
total_frames = 0
distraction_start_time = None
calibration_start_time = None
eye_closed_frames = 0
blink_counter = 0
normal_ear = 0.3  # Default, will be calibrated

# Helper Functions
def distance(p1, p2, w, h):
    # Calculate Euclidean distance between two points
    return math.sqrt(((p1.x * w) - (p2.x * w))**2 + ((p1.y * h) - (p2.y * h))**2)

def eye_aspect_ratio(eye_points, w, h):
    # Calculate Eye Aspect Ratio (EAR) using 6 landmarks
    # Vertical distances (top to bottom of eye)
    vertical1 = distance(eye_points[1], eye_points[5], w, h)
    vertical2 = distance(eye_points[2], eye_points[4], w, h)
    # Horizontal distance (corner to corner)
    horizontal = distance(eye_points[0], eye_points[3], w, h)
    # EAR formula: average vertical / horizontal
    return (vertical1 + vertical2) / (2.0 * horizontal)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while cap.isOpened():
    direction = "Forward"  # Initialize default value
    avg_ear = 0.0
    perclos = 0.0
    is_distracted = False
    is_drowsy = False
    is_yawn = False
    phone_detected = False
    seatbelt_off = False

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process detectors
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    # Calibration mode
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        calibrating = True
        calibration_start_time = time.time()
    
    if calibrating:
        if time.time() - calibration_start_time < CALIBRATION_TIME:
            cv2.putText(frame, f"CALIBRATING... {int(CALIBRATION_TIME - (time.time() - calibration_start_time))}s", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                left_eye = [landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144]]
                right_eye = [landmarks[362], landmarks[385], landmarks[387], landmarks[263], landmarks[373], landmarks[380]]
                ear_left = eye_aspect_ratio(left_eye, w, h)
                ear_right = eye_aspect_ratio(right_eye, w, h)
                normal_ear = (ear_left + ear_right) / 2.0
                EAR_THRESHOLD = normal_ear * 0.7  # Set threshold as 70% of normal EAR
        else:
            if face_results.multi_face_landmarks:
                forward_head_position = face_results.multi_face_landmarks[0].landmark[1].x
            calibrating = False

    # Face analysis
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        
        # Head direction
        if forward_head_position is not None:
            nose = landmarks[1]
            offset = nose.x - forward_head_position
            direction = "Right" if offset > 0.03 else "Left" if offset < -0.03 else "Forward"
        
        # Improved Eye Closure Detection
        left_eye_points = [landmarks[33], landmarks[160], landmarks[158], 
                         landmarks[133], landmarks[153], landmarks[144]]
        right_eye_points = [landmarks[362], landmarks[385], landmarks[387], 
                          landmarks[263], landmarks[373], landmarks[380]]
        
        ear_left = eye_aspect_ratio(left_eye_points, w, h)
        ear_right = eye_aspect_ratio(right_eye_points, w, h)
        avg_ear = (ear_left + ear_right) / 2.0
        
        # Eye state detection
        if avg_ear < EAR_THRESHOLD:
            eye_closed_frames += 1
            if eye_closed_frames >= EAR_CONSEC_FRAMES and not is_drowsy:
                cv2.putText(frame, "EYES CLOSED!", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if sound and not pygame.mixer.get_busy():
                    sound.play()
        else:
            if eye_closed_frames >= EAR_CONSEC_FRAMES:
                blink_counter += 1
                blink_duration = eye_closed_frames / 30  # Assuming 30 FPS
                if blink_duration > 0.5:  # Long blink indicates drowsiness
                    is_drowsy = True
            eye_closed_frames = 0
        
        # PERCLOS calculation
        total_frames += 1
        if avg_ear < EAR_THRESHOLD:
            closed_frames += 1
        perclos = closed_frames / total_frames
        
        # Yawn detection (improved)
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        mouth_left = landmarks[308]
        mouth_right = landmarks[78]
        mouth_height = distance(mouth_top, mouth_bottom, w, h)
        mouth_width = distance(mouth_left, mouth_right, w, h)
        mar = mouth_height / mouth_width
        if mar > YAWN_THRESHOLD:
            is_yawn = True

    # Enhanced Phone Detection
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Get key landmarks
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            wrist = hand_landmarks.landmark[0]
            
            # Calculate distances
            pinch_distance = distance(thumb_tip, index_tip, w, h)
            hand_to_face_distance = distance(wrist, landmarks[1], w, h) if face_results.multi_face_landmarks else float('inf')
            
            # Phone detection conditions
            is_pinching = pinch_distance < PHONE_DISTANCE_THRESHOLD * w
            hand_near_face = hand_to_face_distance < HAND_FACE_THRESHOLD * w
            
            # Finger state detection
            fingers_up = 0
            for finger_tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                if hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[pip].y:
                    fingers_up += 1
            
            # Phone detection logic
            if is_pinching and hand_near_face and 1 <= fingers_up <= 2:
                phone_detected = True
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

    # Seatbelt detection
    if pose_results.pose_landmarks:
        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if left_shoulder.y > SEATBELT_HEIGHT_THRESHOLD or right_shoulder.y > SEATBELT_HEIGHT_THRESHOLD:
            seatbelt_off = True

    # Distraction logic
    is_distracted = phone_detected
    if face_results.multi_face_landmarks:
        is_distracted = is_distracted or (direction != "Forward")
    elif distraction_start_time is None:
        distraction_start_time = time.time()
    elif time.time() - distraction_start_time > DISTRACTION_THRESHOLD:
        is_distracted = True

    # Alert system
    alerts = []
    if is_distracted:
        alerts.append(("DISTRACTED!", (50, 50), (0, 0, 255)))
    if is_drowsy:
        alerts.append(("DROWSY!", (50, 100), (0, 0, 255)))
    if is_yawn:
        alerts.append(("YAWN DETECTED", (50, 150), (0, 140, 255)))
    if seatbelt_off:
        alerts.append(("SEATBELT OFF!", (w - 250, 50), (0, 0, 255)))

    for text, pos, color in alerts:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if sound:
        if any([is_distracted, is_drowsy]) and not pygame.mixer.get_busy():
            sound.play()
        elif not any([is_distracted, is_drowsy]):
            sound.stop()

    # Display info
    cv2.putText(frame, f"Head: {direction}", (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Blinks: {blink_counter}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Press 'C' to calibrate", (w - 200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Advanced Driver Monitor", frame)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()