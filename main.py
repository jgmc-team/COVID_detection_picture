# -*- coding: utf-8 -*-
from sklearn.metrics import f1_score
import csv


def csv_reader(file_obj):
    """
    Read a csv file
    """
    reader = csv.reader(file_obj)
    # next(reader, None)
    labels = []
    for row in reader:
        # print(row)
        labels.append(int(row[1]))
    return labels


def load_data(csv_path):
    with open(csv_path, "r") as f_obj:
        labels = csv_reader(f_obj)

    return labels


def main():
    labels_true = load_data('data_validate.csv')
    labels_pred = load_data('data_decision.csv')

    try:
        score = f1_score(labels_true, labels_pred, average="weighted")
        print('f1-score:', score)
    except Exception as e:
        print('Ошибка:', e)

    file = open("score.txt", "w")
    file.write(str(score))
    file.close()


if __name__ == '__main__':
    main()
