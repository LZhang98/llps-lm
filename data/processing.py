import csv
import numpy as np

def import_data():
    high = []
    with open('high.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            high.append(row)
    low = []
    with open('low.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            low.append(row)
    return high, low

if __name__ == '__main__':
    print('processing.py')
    high, low = import_data()
    print(len(high), len(low))