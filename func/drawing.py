import cv2 as open_cv


COLOR_RED = (255, 0, 0)

def draw_parking_spot(image,
                  coordinates,
                  label,
                  font_color,
                  border_color=COLOR_RED,
                  line_thickness=1,
                  font=open_cv.FONT_HERSHEY_SIMPLEX,
                  font_scale=0.5):
    '''
    :param image:
    :param coordinates:
    :param label:
    :param font_color:
    :param border_color:
    :param line_thickness:
    :param font:
    :param font_scale:
    :return:
    '''
    open_cv.drawContours(image,
                         [coordinates],
                         contourIdx=-1,
                         color=border_color,
                         thickness=2,
                         lineType=open_cv.LINE_8)
    moments = open_cv.moments(coordinates)

    center = (int(moments["m10"] / moments["m00"]) - 3,
              int(moments["m01"] / moments["m00"]) + 3)

    open_cv.putText(image,
                    label,
                    center,
                    font,
                    font_scale,
                    font_color,
                    line_thickness,
                    open_cv.LINE_AA)