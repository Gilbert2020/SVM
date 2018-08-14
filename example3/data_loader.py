import csv
import numpy as np


def data_loader(path="data.csv"):

    x = np.zeros((10000,2))
    y = np.zeros(10000)

    with open(path, 'r', encoding='utf8') as csvfile:
        r = csv.reader(csvfile, delimiter=',')

        for idx, row in enumerate(r):

            try:
                x[idx, 0] = row[0]
                x[idx, 1] = row[1]

                if row[2] == "False":
                    y[idx] = 0
                else:
                    y[idx] = 1
            except IndexError:
                continue


    return x, y