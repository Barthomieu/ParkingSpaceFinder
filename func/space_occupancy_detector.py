import cv2 as open_cv
import numpy as np
import logging
from func.drawing import draw_parking_spot


COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)

class SpotOccupancy:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []

    def detect_motion(self):
        capture = open_cv.VideoCapture(self.video)
        capture.set(open_cv.CAP_PROP_POS_FRAMES, self.start_frame)

        coordinates_data = self.coordinates_data
        logging.debug("coordinates data: %s", coordinates_data)
        print("coordinates data: %s", coordinates_data)
        # coordinates data: % s[{'id': 0, 'coordinates': [[145, 310], [204, 313], [222, 354], [153, 365]]}, {'id': 1, 'coordinates': [
        # [361, 322], [415, 319], [437
        #    , 343], [394, 348]]}, {'id': 2, 'coordinates': [[296, 444], [368, 451], [340, 473], [303, 458]]}]

        for p in coordinates_data:
            coordinates = self._coordinates(p)
            logging.debug("coordinates: %s", coordinates)
            print("coordinates: %s", coordinates)
            #coordinates: % s[[145 310]
            #[204 313]
            #[222 354]
            #[153 365]]
            rect = open_cv.boundingRect(coordinates) #calcutate top-up bounging ractingle of set point. return x, y, w, h
            logging.debug("rect: %s", rect)
            print("rect: %s", rect)
            #rect: %s (589, 281, 115, 48)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0] # punkty x zaznacznone - punkty x z boundig boxa
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1] #y
            logging.debug("new_coordinates: %s", new_coordinates)

            print("new_coordinates: %s", new_coordinates, coordinates[:, 0],coordinates[:, 1])

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
            #print("mask: %s", self.mask)

        statuses = [False] * len(coordinates_data)
        times = [None] * len(coordinates_data)

        while capture.isOpened():
            sucess, frame = capture.read()
            if frame is None:
                break

            if not sucess:
                raise CaptureReadError("Error reading video capture on frame %s" % str(frame))

            blurred = open_cv.GaussianBlur(frame.copy(), (5, 5), 3)
            grayed = open_cv.cvtColor(blurred, open_cv.COLOR_BGR2GRAY)
            new_frame = frame.copy()
            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(open_cv.CAP_PROP_POS_MSEC) / 1000.0
            print("pos in seconds", position_in_seconds)
            for index, c in enumerate(coordinates_data):
                print(index, c)
                status = self.__apply(grayed, index, c)

                if times[index] is not None and self.same_status(statuses, index, status):
                    times[index] = None
                    continue

                if times[index] is not None and self.status_changed(statuses, index, status):
                    if position_in_seconds - times[index] >= SpotOccupancy.DETECT_DELAY:
                        statuses[index] = status
                        times[index] = None
                    continue

                if times[index] is None and self.status_changed(statuses, index, status):
                    times[index] = position_in_seconds

            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)

                color = COLOR_GREEN if statuses[index] else COLOR_BLUE
                draw_parking_spot(new_frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)

            open_cv.imshow(str(self.video), new_frame)
            k = open_cv.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        open_cv.destroyAllWindows()

    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)

        rect = self.bounds[index]
        logging.debug("rect: %s", rect)

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = open_cv.Laplacian(roi_gray, open_cv.CV_64F)
        logging.debug("laplacian: %s", laplacian)

        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        status = np.mean(np.abs(laplacian * self.mask[index])) < SpotOccupancy.LAPLACIAN
        logging.debug("status: %s", status)

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