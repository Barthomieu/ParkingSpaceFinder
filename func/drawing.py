import cv2 as open_cv
from func.utils import k_closest, midpoint

RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255,255,0)
font=open_cv.FONT_HERSHEY_SIMPLEX

def draw_parking_spot(image,
                  coordinates,
                  label,
                  font_color,
                spot_counter=0,
                  border_color=RED,
                  line_thickness=1,
                  font=open_cv.FONT_HERSHEY_SIMPLEX,
                  font_scale=0.5):
    '''
    :param image: the image on which the frames will be applied
    :param coordinates: coordinates of marked parking spot
    :param label: label of detected object
    :param font_color:
    :param border_color:
    :param line_thickness:
    :param font:
    :param font_scale:
    :return: returns an image with marked parking spaces
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

    open_cv.putText(image,
                    "Wolne miejsca: " + str(spot_counter),
                    (50, 50),
                    font,
                    font_scale,
                    font_color,
                    line_thickness,
                    open_cv.LINE_AA)



def draw_lines_between_cars(frame, centroids, vehicle_size ):
    closest_points = []
    print("CENTROIDS \n", centroids )
    for point in centroids:
        print("Najbliższe punnkty ", point, k_closest(centroids,point,3))

        closest_points.append( k_closest(centroids,point,3))

    sum_of_spot = 0

    for p in closest_points:
        print("wynik z k_closest", p)
        print("tuple", tuple(p[0]), tuple(p[1]),"direction", p[1][3], p[2][3],  "lista", p[0],p[1], p[1][2])
        p1 = (int(p[0][0]),int(p[0][1]))
        p2 = (int(p[1][0]),int(p[1][1]))
        p3 = (int(p[2][0]),int(p[2][1]))

        if p[1][2] > (vehicle_size*1.5):
            center = midpoint(p1, p2)
            count = round(p[1][2] / (vehicle_size*1.3))
            frame = open_cv.line(frame,p1, p2, GREEN,4 )
            frame = open_cv.putText(frame, str(count), center, font, 1, YELLOW, 2, open_cv.LINE_AA)
            sum_of_spot+=count


        if p[2][2] > (vehicle_size*1.5) and p[1][3] != p[2][3]:
            center = midpoint(p1, p3)
            count = round(p[2][2] / (vehicle_size*1.3))

            frame = open_cv.line(frame, p1, p3, GREEN, 4)
            frame = open_cv.putText(frame, str(count), center, font, 1, YELLOW, 2, open_cv.LINE_AA)
            sum_of_spot += count
#        else:
            #frame = open_cv.line(frame, p1, p3, RED, 4)
    #open_cv.putText(frame, "Wolne miejsca " + str(sum_of_spot), (50,100), font, 2, RED, 2, open_cv.LINE_AA)
            #frame = open_cv.line(frame, (int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), RED, 2)
    return frame

