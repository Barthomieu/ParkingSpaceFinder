import cv2
import torch
import numpy

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

model = torch.hub.load(r'C:\Users\Bartłomiej\PycharmProjects\ParkingSpaceFinder\yolov5', 'custom', path=r'C:\Users\Bartłomiej\PycharmProjects\ParkingSpaceFinder\yolov5s.pt', source='local', force_reload=True)

img = cv2.imread('images/zdj6.png')
# Inference
results = model(img, size=640)  # includes NMS
# Results
results.print()
results.save()
print(results.xyxy[0] ) # im predictions (tensor)
df = results.pandas().xyxy[0]
print(df.head())
labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
print("labels" )
print(labels)
print("cords" )
print(cord)


df['center_x']= df[['xmin','xmax']].mean(axis=1)
df['center_y']= df[['ymin','ymax']].mean(axis=1)
print(df.head())