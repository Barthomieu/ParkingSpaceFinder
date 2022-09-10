import cv2 as open_cv
import numpy as np
from func.drawing import draw_parking_spot

COLOR_WHITE = (255, 255, 255)


class Coordinates:
    KEY_RESET= ord('r')
    KEY_QUIT= ord('q')


    def __init__(self, img, output, color):
        '''
        :param img:
        :param output:
        '''
        self.output = output
        self.caption = img
        self.color = color
        self.image = open_cv.imread(img).copy()
        self.click_count = 0
        self.ids = 0
        self.coordinates = []

        open_cv.namedWindow(self.caption, open_cv.WINDOW_GUI_EXPANDED)
        open_cv.setMouseCallback(self.caption, self.__mouse_callback)

    def generate(self):
        '''
        :return:
        '''
        while True:
            open_cv.imshow(self.caption, self.image)
            key = open_cv.waitKey(0)

            if key == Coordinates.KEY_RESET:
                self.image = self.image.copy()
            elif key == Coordinates.KEY_QUIT:
                break
        open_cv.destroyWindow(self.caption)

    def __mouse_callback(self, event, x, y, flags, params):
        '''
        :param event:
        :param x:
        :param y:
        :param flags:
        :param params:
        :return:
        '''

        if event == open_cv.EVENT_LBUTTONDOWN:
            self.coordinates.append((x, y))
            self.click_count += 1

            if self.click_count >= 4:
                self.__handle_done()

            elif self.click_count > 1:
                self.__handle_click_progress()

        open_cv.imshow(self.caption, self.image)

    def __handle_click_progress(self):
        open_cv.line(self.image, self.coordinates[-2], self.coordinates[-1], (255, 0, 0), 1)

    def __handle_done(self):
        open_cv.line(self.image,
                     self.coordinates[2],
                     self.coordinates[3],
                     self.color,
                     1)
        open_cv.line(self.image,
                     self.coordinates[3],
                     self.coordinates[0],
                     self.color,
                     1)

        self.click_count = 0

        coordinates = np.array(self.coordinates)

        self.output.write("-\n          id: " + str(self.ids) + "\n          coordinates: [" +
                          "[" + str(self.coordinates[0][0]) + "," + str(self.coordinates[0][1]) + "]," +
                          "[" + str(self.coordinates[1][0]) + "," + str(self.coordinates[1][1]) + "]," +
                          "[" + str(self.coordinates[2][0]) + "," + str(self.coordinates[2][1]) + "]," +
                          "[" + str(self.coordinates[3][0]) + "," + str(self.coordinates[3][1]) + "]]\n")

        draw_parking_spot(self.image, coordinates, str(self.ids + 1), COLOR_WHITE)

        for i in range(0, 4):
            self.coordinates.pop()

        self.ids += 1