import mediapipe as mp
import numpy as np
import tensorflow as tf
import cv2
import time

# Set the desired width and height for the video feed
w, h = 640, 480

# Define the actions for action recognition
actions = {0: "Sitting", 1: "Standing", 2: "Walking"}


# Create a PoseDetector class
class PoseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.9, trackCon=0.5):

        # Initialize parameters for the MediaPipe Holistic model
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe components
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(self.mode, self.upBody, self.smooth, False, False, False,
                                                 self.detectionCon, self.trackCon)
        self.drawSpec = self.mpDraw.DrawingSpec((255, 0, 0), thickness=1, circle_radius=1)

        # Load the trained action recognition model
        self.new_model = tf.keras.models.load_model('model_mediapipe/action1_1.keras')
        self.sequence = []
        self.sentence = []

    def findPose(self, img, draw=True):

        # Convert image to RGB format
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(imgRGB)

        # Draw landmarks on the image
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        if results.pose_landmarks:
            if draw:
                # self.mpDraw.draw_landmarks(img, results.face_landmarks, self.mpHolistic.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                self.mpDraw.draw_landmarks(img, results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)
                self.mpDraw.draw_landmarks(img, results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS)

        return results, img

    def extract_keypoints(self, results):
        # Extract and flatten pose landmarks
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        return np.concatenate([pose])


# Define the main function
def main():
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    while True:
        # Read frames from the camera
        success, img = cap.read()
        results, img = detector.findPose(img)

        # Extract keypoints and predict actions
        keypoints = detector.extract_keypoints(results)
        detector.sequence.append(keypoints)
        detector.sequence = detector.sequence[-30:]

        if len(detector.sequence) == 30:
            res = detector.new_model.predict(np.expand_dims(detector.sequence, axis=0))[0]
            class_idx = int(np.argmax(res))
            cv2.putText(img, actions[class_idx], (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Display the action recognition output
        cv2.imshow("Action Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
