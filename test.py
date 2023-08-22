from ultralytics import YOLO
import cv2
model = YOLO("best.pt")
result = model.predict("yibiaopan.jpg",save=True)
# print("结果长度",len(result))
print(result[0].boxes)
for label in result[0].boxes.cls:
    print(label)
    print(model.names[int(label)])

image = cv2.imread("yibiaopan.jpg")
h,w,d = image.shape

for xyxy in result[0].boxes.xyxy:
    x1,y1,x2,y2 = xyxy
    cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))

cv2.imshow("",image)
cv2.waitKey()