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

def deephase_data(): # no features

    # [(label, sequence)]
    data = []
    # [LLPS+/-]
    categories = []
    with open('./data/training_data_features.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            data.append((row[1], row[0]))
            categories.append(row[2])

    return data, categories

if __name__ == '__main__':
    print('processing.py')
    high, low = import_data()
    print(len(high), len(low))