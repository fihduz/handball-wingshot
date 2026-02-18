import cv2
import mediapipe as mp
import mediapipe.solutions as mps

mp_drawing = mps.drawing_utils
mp_pose = mps.pose

cap = cv2.VideoCapture(path)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break