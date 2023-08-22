from ultralytics import YOLO
import cv2
model = YOLO("yolov8n.pt")
model.train(data="electricbicycle.yaml",cfg="args.yaml")
