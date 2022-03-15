import os
import csv
import time
import zmq
import json
import sys

os.chdir(sys._MEIPASS)
try:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    while True:
        with open('sample-detections.csv', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    message = socket.recv()
                    print(f"Received request: {message}")
                    time.sleep(0.1)
                    c = row['class']
                    x = row['x']
                    y = row['y']
                    z = row['z']
                    result = [c,x,y,z]
                    json_string = json.dumps(result)
                    socket.send_string(json_string)
finally:
    context.term()


