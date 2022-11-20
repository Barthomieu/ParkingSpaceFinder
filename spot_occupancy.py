import cv2 as open_cv
import numpy as np
import logging
from func.drawing import draw_parking_spot
import torch
import json
from matplotlib import path

BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

class SpotOccupancy:
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.model = self.load_model()
        self.contours = []
        self.bounds = []
        self.mask = []


    def load_model(self):
        """
       load yolov5 model with pytorch hub
        :return: pre-trained model
        """
        model = torch.hub.load(r'C:\Users\Bartłomiej\PycharmProjects\ParkingSpaceFinder\yolov5', 'custom', path=r'C:\Users\Bartłomiej\PycharmProjects\ParkingSpaceFinder\yolov5s.pt', source='local', force_reload=True)
        model.conf = 0.25  # confidence threshold (0-1)
        model.iou = 0.45  # NMS IoU threshold (0-1)
        model.classes = [2,3,5,7]  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

        return model

    def draw_centroids_on_image(self, output_image, json_results):
        data = json.loads(json_results)  # Converting JSON array to Python List
        # Accessing each individual object and then getting its xmin, ymin, xmax and ymax to calculate its centroid
        centroids_list = []
        for objects in data:
            #DANE Z YOLO {'xmin': 280.4359130859, 'ymin': 226.7133026123, 'xmax': 321.1462402344, 'ymax': 256.8363342285, 'confidence': 0.372656852, 'class': 2, 'name': 'car'}

            xmin = objects["xmin"]
            ymin = objects["ymin"]
            xmax = objects["xmax"]
            ymax = objects["ymax"]

            # Centroid Coordinates of detected object
            cx = int((xmin + xmax) / 2.0)
            cy = int((ymin + ymax) / 2.0)
            centroids = [cx,cy]
            centroids_list.append(centroids)

            gray = open_cv.cvtColor(output_image, open_cv.COLOR_BGR2GRAY)
            thresh = open_cv.threshold(gray, 0, 255, open_cv.THRESH_BINARY_INV + open_cv.THRESH_OTSU)[1]
            cnts = open_cv.findContours(thresh, open_cv.RETR_EXTERNAL, open_cv.CHAIN_APPROX_SIMPLE)


            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            print("CNTS", cnts)
            rect = open_cv.minAreaRect(cnts[0])
            print("RECT", rect)
            box = np.int0(open_cv.boxPoints(rect))
            print("BOX", box)
            open_cv.drawContours(output_image, [box], 0, (36, 255, 12), 3)


            open_cv.circle(output_image, (cx, cy), 2, (0, 0, 255), 2, open_cv.FILLED)  # draw center dot on detected object
            #open_cv.putText(output_image, str(str(cx) + " , " + str(cy)), (int(cx) - 40, int(cy) + 30),
                       # open_cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, open_cv.LINE_AA)

        return (output_image, centroids_list)


    def detect_car_on_marked_spot(self):
        capture = open_cv.VideoCapture(self.video)
        capture.set(open_cv.CAP_PROP_POS_FRAMES, self.start_frame)

        coordinates_data = self.coordinates_data
        #coordinates data: [{'id': 0, 'coordinates': [[145, 310], [204, 313], [222, 354], [153, 365]]},
        #                   {'id': 1, 'coordinates': [[361, 322], [415, 319], [437, 343], [394, 348]]},
        #                   {'id': 2, 'coordinates': [[296, 444], [368, 451], [340, 473], [303, 458]]}]

        marked_spots = len(coordinates_data)  # liczba zaznaczonych miejsc parkingowych
        free_spots = 0  # liczba wolnych miejsc parkingowych

        for p in coordinates_data:
            coordinates = self._coordinates(p)
            rect2 = open_cv.minAreaRect(coordinates)
            rect = open_cv.boundingRect(coordinates)
            logging.debug("rect: %s", rect)
            print("rect", rect)
            print("rect 2 ____", rect2)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]
            logging.debug("new_coordinates: %s", new_coordinates)

            """
            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = open_cv.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=open_cv.LINE_8)

            mask = mask == 255
            self.mask.append(mask)
            logging.debug("mask: %s", self.mask)
        """
        statuses = [False] * len(coordinates_data)
        times = [None] * len(coordinates_data)

        while capture.isOpened():
            sucess, frame = capture.read()
            if frame is None:
                break

            if not sucess:
                raise CaptureReadError("Error reading video capture on frame %s" % str(frame))


            #zmiana parametrów obrazy dla poprawy detekcji
            #blurred = open_cv.GaussianBlur(frame.copy(), (5, 5), 3)
            #grayed = open_cv.cvtColor(blurred, open_cv.COLOR_BGR2GRAY)

            results = self.model(frame)
            json_results = results.pandas().xyxy[0].to_json(orient="records")
            #print(json_results)

            #points_data = results.pandas().xyxy[0]
            #points_data['center_x'] = points_data[['xmin', 'xmax']].mean(axis=1)
            #points_data['center_y'] = points_data[['ymin', 'ymax']].mean(axis=1)

            new_frame = frame.copy() # kopia do wyświetlania na końcowym ekranie

            #dodanie centroidów do kadru
            frame2, centroids = self.draw_centroids_on_image(frame, json_results)

            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(open_cv.CAP_PROP_POS_MSEC) / 1000.0

            for index, c in enumerate(coordinates_data):
                status = self.check_status( centroids, index, c)  # sprawdzanie status miejsca parkingowego

                if times[index] is not None and self.same_status(statuses, index, status):
                    times[index] = None
                    continue

                if times[index] is not None and self.status_changed(statuses, index, status):
                    #if position_in_seconds - times[index] >= SpotOccupancy.DETECT_DELAY:
                    statuses[index] = status
                    times[index] = None
                    continue

                if times[index] is None and self.status_changed(statuses, index, status):
                    times[index] = position_in_seconds

            for index, p in enumerate(coordinates_data): # zmiana koloru miejsca parkingowego
                coordinates = self._coordinates(p)
                occupated_spots = sum(statuses) #zlicza ilość TRUE w liście statusow
                free_spots = len(coordinates_data)-occupated_spots
                color = BLUE if statuses[index] else GREEN
                draw_parking_spot(frame2, coordinates, str(p["id"] + 1), WHITE, free_spots, color)

            #print("statusy", statuses)

            open_cv.imshow(str(self.video), frame2)
            k = open_cv.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        open_cv.destroyAllWindows()

    def pointInRect(self, cx, cy, rect):
        x1, y1, w, h = rect
        x2, y2 = x1 + w, y1 + h
        if (x1 < cx and cx < x2):
            if (y1 < cy and cy < y2):
                return True
        return False

    def point_in_polygon(self, coordinates, list_of_centroids):
        p = path.Path(coordinates)
        result_list = p.contains_points(list_of_centroids)

        if True in result_list:
            return True

        return False


    def check_status(self, centroids_list, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)
        """
        rect = self.bounds[index]
        #print("rectr", rect , centroids_data)
        logging.debug("rect: %s", rect)
        #print("sprawdzam status", center_x,center_y)
        for i, row in centroids_data.iterrows():
            #print("POINT", row)
            center_x = row["center_x"]
            center_y = row["center_y"]
            status = self.pointInRect(center_x, center_y, rect)
            if status == False:
                return status
        logging.debug("status: %s", status)
            """
        status = self.point_in_polygon(coordinates, centroids_list)

        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])

    @staticmethod
    def same_status(coordinates_status, index, status):
        return status == coordinates_status[index]

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]


class CaptureReadError(Exception):
    pass


#(x0, y0) = ((b2-b1)/(a1-a2), a2*(b2-b1)/(a1-a2)+b2)