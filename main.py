import argparse
import yaml
from func.mark_coordinates import Coordinates
from func.space_occupancy_detector import SpotOccupancy
import logging

COLOR_RED = (255, 0, 0)

def main():
    logging.basicConfig(level=logging.DEBUG)

    args = parse_args()

    image_file = args.image_config
    data_file = args.coords_file
    start_frame = args.start_frame

    if image_file is not None:
        with open(data_file, "w+") as points:
            generator = Coordinates(image_file, points, COLOR_RED)
            generator.generate()

    with open(data_file, "r") as data:
        points = yaml.load(data, yaml.FullLoader)
        detector = SpotOccupancy(args.video_path, points, int(start_frame))
        detector.detect_motion()

def parse_args():
    parser = argparse.ArgumentParser(description='Generates Coordinates File')

    parser.add_argument("--image",
                        dest="image_config",
                        required=False,
                        help="Image file to generate coordinates on")

    parser.add_argument("--video",
                        dest="video_path",
                        required=True,
                        help="Video file to detect motion on")

    parser.add_argument("--data",
                        dest="coords_file",
                        required=True,
                        help="Data file to be used with OpenCV")

    parser.add_argument("--start-frame",
                        dest="start_frame",
                        required=False,
                        default=1,
                        help="Starting frame on the video")

    return parser.parse_args()


if __name__== '__main__':
    main()