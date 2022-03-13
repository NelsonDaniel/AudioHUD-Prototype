import os
import csv
import time

with open(os.path.join(os.path.dirname(__file__),'sample-detections.csv'), newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        time.sleep(0.1)
        c = row['class']
        x = row['x']
        y = row['y']
        z = row['z']
        if c != 'NOTHING':
            print(c, x, y, z)