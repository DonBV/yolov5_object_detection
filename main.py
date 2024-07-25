import sys
import os
import cv2
import torch
import numpy as np
import streamlink

# Add yolov5 directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# Load YOLOv5 model
device = select_device('')
model = DetectMultiBackend('yolov5/yolov5s.pt', device=device, dnn=False, data=None, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
img_size = (640, 640)

# Function to detect objects
def detect_objects(frame):
    img = cv2.resize(frame, (img_size[1], img_size[0]))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    det = pred[0]
    if len(det):
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)
            # Print detailed information about each object
            print(f"Detected object: {names[int(cls)]}, coordinates: {xyxy}, confidence: {conf:.2f}")

    return frame

def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=3):
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    if label:
        tf = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main():
    # Open video stream using Streamlink
    twitch_url = 'https://www.twitch.tv/channel_name'
    streams = streamlink.streams(twitch_url)
    if 'best' in streams:
        stream_url = streams['best'].url
    else:
        print("Error: Unable to get video stream.")
        return

    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            break

        # Detect objects
        frame = detect_objects(frame)

        # Display frame with detected objects
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
