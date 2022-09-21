import cv2
import torch
import pandas
import numpy
import json
# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

model = torch.hub.load(r'C:\Users\Bartłomiej\PycharmProjects\ParkingSpaceFinder\yolov5', 'custom', path=r'C:\Users\Bartłomiej\PycharmProjects\ParkingSpaceFinder\yolov5s.pt', source='local', force_reload=True)

model.conf = 0.25  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
model.classes = [2]  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs


'''
img = cv2.imread('images/zdj6.png')
video = cv2.VideoCapture('video/test1.mp4')
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
'''
def draw_centroids_on_image(output_image, json_results):
    data = json.loads(json_results)  # Converting JSON array to Python List
    # Accessing each individual object and then getting its xmin, ymin, xmax and ymax to calculate its centroid
    for objects in data:
        xmin = objects["xmin"]
        ymin = objects["ymin"]
        xmax = objects["xmax"]
        ymax = objects["ymax"]

        # print("Object: ", data.index(objects))
        # print ("xmin", xmin)
        # print ("ymin", ymin)
        # print ("xmax", xmax)
        # print ("ymax", ymax)

        # Centroid Coordinates of detected object
        cx = int((xmin + xmax) / 2.0)
        cy = int((ymin + ymax) / 2.0)
        # print(cx,cy)

        cv2.circle(output_image, (cx, cy), 2, (0, 0, 255), 2, cv2.FILLED)  # draw center dot on detected object
        cv2.putText(output_image, str(str(cx) + " , " + str(cy)), (int(cx) - 40, int(cy) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return (output_image, cx, cy)

def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            #cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame

def score_frame( frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    :param frame: input frame in numpy/list/tuple format.
    :return: Labels and Coordinates of objects detected by model in the frame.
    """

    frame = [frame]
    results = model(frame)

    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

print("+++++++++++=results by cv2")
cap = cv2.VideoCapture('video/test1.mp4')
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    sucess, frame = cap.read()
    if not sucess:
        break
    results = model(frame)
    cv2.imshow("before", frame)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV image (BGR to RGB)

    #results = score_frame(image)
    #results.xyxy[0]  # im predictions (tensor)
    #results.pandas().xyxy[0]  # im predictions (pandas)
    json_results = results.pandas().xyxy[0].to_json(orient="records")

    #frame = plot_boxes(results, image)
    frame, cx, cy = draw_centroids_on_image(frame, json_results)
    cv2.imshow("after", frame)

    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


