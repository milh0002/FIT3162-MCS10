import numpy as np
import cv2
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression
import tensorflow as tf

w, h = 640, 512
actions = {0: "Sitting", 1: "Standing", 2: "Walking"}

class PoseDetector:
    def __init__(self, poseweights="yolov7-w6-pose.pt"):
        self.device = select_device('cpu')  # Change to GPU if available

        self.model = attempt_load(poseweights, map_location=self.device)
        self.model.eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # YOLO class names

        # Load the Keras action recognition model
        self.load_keras_action_model()

        self.sequence = []

    def load_keras_action_model(self):
        self.new_model = tf.keras.models.load_model('model_yolo/action_cnn_model.keras')

    def extract_keypoints(self, output_data):
        keypoints_single_frame = []

        for output in output_data[0]:
            xyxy, conf, cls, kpts = output[:4], output[4], output[5], output[6:]
            c = int(cls)
            keypoints = kpts  # Extract keypoints from the detection
            keypoints_single_frame.append(keypoints)

        return [keypoints_single_frame]

    def draw_keypoints(self, output_data, img):
        for output in output_data[0]:
            xyxy, conf, cls, kpts = output[:4], output[4], output[5], output[6:]
            c = int(cls)
            keypoints = kpts  # Extract keypoints from the detection
            keypoints = keypoints.reshape(-1, 3)

            for kp in keypoints:
                x, y, score = kp
                if score > 0.3:  # Adjust the score threshold as needed
                    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)  # Draw keypoints

        return img

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = letterbox(imgRGB, (w, h), stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        with torch.no_grad():
            output_data, _ = self.model(image)

        output_data = non_max_suppression(output_data, conf_thres=0.25, iou_thres=0.45)

        im0 = image[0].permute(1, 2, 0) * 255
        im0 = im0.cpu().numpy().astype(np.uint8)
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

        keypoints = self.extract_keypoints(output_data)

        if draw:
            im0 = self.draw_keypoints(output_data, im0)

        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        if len(self.sequence) == 30:
            res = self.new_model.predict(np.expand_dims(self.sequence, axis=0))[0]
            class_idx = int(np.argmax(res))
            cv2.putText(im0, actions[class_idx], (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)

        return im0

def main():
    detector = PoseDetector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        cv2.imshow("YOLOv7 Action Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
