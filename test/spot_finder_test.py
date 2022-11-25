
import yaml
from func.free_spot_finder import SpotFinder
from func.mark_coordinates import Coordinates

RED = (255, 0, 0)

image_file = '../images/vid2_frame.jpg'
data_file = '../coord/coordinates_1.yml'

video_path = '../video/vid2.mp4'

if image_file is not None:
    with open(data_file, "w+") as points:
        generator = Coordinates(image_file, points, RED)
        generator.generate()

with open(data_file, "r") as data:
    points = yaml.load(data, yaml.FullLoader)
    finder = SpotFinder(video_path, points )
    finder.calculate_dist_btw_cars()