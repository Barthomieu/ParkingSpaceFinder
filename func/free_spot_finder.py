import imutils
import torch
import cv2 as open_cv
import numpy as np
import json
from matplotlib import path
from func.utils import k_closest, calculate_averange_vehicle_size
from func.drawing import draw_lines_between_cars


class SpotFinder():

    def __init__(self, video, coordinates):
        self.video = video
        self.coordinates_data = coordinates
        self.model = self.load_model()


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



    #zwraca wspolrzedne samochodów wykrytych w zaznaczonym obszarze
    def points_in_polygon(self, coordinates, list_of_centroids):
        p = path.Path(coordinates)
        result_list = p.contains_points(list_of_centroids)

        #wyciągnięcie centroidow z zaznaczonego obszaru
        centroids_inside_area = np.array(list_of_centroids)[np.array(result_list)]
        centroids = list(map(tuple, centroids_inside_area))
        return centroids



    def calculate_dist_btw_cars(self):
        capture = open_cv.VideoCapture(self.video)
        capture.set(open_cv.CAP_PROP_POS_FRAMES, 100)
        coordinates_data = self.coordinates_data
        # coordinates data: [{'id': 0, 'coordinates': [[145, 310], [204, 313], [222, 354], [153, 365]]},
        #                   {'id': 1, 'coordinates': [[361, 322], [415, 319], [437, 343], [394, 348]]},
        #                   {'id': 2, 'coordinates': [[296, 444], [368, 451], [340, 473], [303, 458]]}]
        print("jestem w funcji, sciezka do wideo", self.video)

        while capture.isOpened():
            sucess, frame = capture.read()
            if frame is None:
                break
            open_cv.imshow("test1", frame)
            print("ok jestem w pętli")
            if not sucess:
                raise CaptureReadError("Error reading video capture on frame %s" % str(frame))


            open_cv.imshow("obraz przed modelem", frame)
            results = self.model(frame) # wynik detekcji obiektów za pomocą modelu yolo

            print("uruchomiłem model")
            json_results = results.pandas().xyxy[0].to_json(orient="records")



            print("json RESULTS\n", json_results)
            #results = results.pandas().xyxy[0]
            results = results.pandas().xywh[0]
            #results["cx"] = round((results["xmin"] + results["xmax"]) / 2.0, 2)
            #results["cy"] = round((results["ymin"] + results["ymax"]) / 2.0, 2)
            results["centroids"] = list(zip(results.xcenter, results.ycenter))

            print("RESULTS\n", results )
            all_centroids = results["centroids"].tolist()
            #[{"xmin":239.8869171143,"ymin":114.231048584,"xmax":364.1714172363,"ymax":193.0829162598,"confidence":0.8601888418,"class":2,"name":"car"},
            # {"xmin":353.5339660645,"ymin":236.8017578125,"xmax":452.945098877,"ymax":340.435546875,"confidence":0.8452271223,"class":2,"name":"car"}

            #dodanie centroidów do kadru
            #frame2, all_centroids = self.draw_centroids_on_image(frame, json_results)


            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)
                centroids_in_polygon = self.points_in_polygon(coordinates, all_centroids)
                print("Centorids in polyhon\n", centroids_in_polygon)
                #wyciagam dane samochodow znajdujacych sie w zaznaczonych miejscach
                cars_in_area = results[results["centroids"].isin(centroids_in_polygon)]
                print("cars in area/n", cars_in_area)
                centroids = cars_in_area["centroids"].tolist()
                vehicle_size = calculate_averange_vehicle_size(coordinates, cars_in_area )
                print("vehicle size", vehicle_size)

                frame2 = draw_lines_between_cars(frame, centroids,vehicle_size)
                frame2 = imutils.resize(frame2, width=800)
                #print("list of closest points", list_of_closest_points)

            open_cv.imshow(str(self.video), frame2)
            k = open_cv.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        open_cv.destroyAllWindows()



    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])

class CaptureReadError(Exception):
    pass