import cv2
import requests
import numpy as np
import time

# URL of the live stream
url = "http://206.127.78.103:80/GetData.cgi?CH=1"

while True:
    # Get the HTML content of the URL
    response = requests.get(url)

    # Find the source of the video stream
    start_index = response.text.find('src="') + 5
    end_index = response.text.find('"', start_index)
    video_url = response.text[start_index:end_index]

    # Capture the current frame from the video stream
    cap = cv2.VideoCapture(video_url)
    ret, frame = cap.read()

    # Save the current frame to a file
    cv2.imwrite("current_frame.jpg", frame)

    # Release the capture and wait for 10 seconds
    cap.release()
    time.sleep(10)
