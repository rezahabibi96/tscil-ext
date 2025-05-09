import wget
import numpy as np
import os
import pickle
from pandas import read_csv
from sklearn.model_selection import train_test_split
from zipfile import ZipFile


def load_file(filepath):
    dataframe = read_csv(filepath, header=None)
    return dataframe.values


def load_sub(path):
    X_sub = []

    for seg in range(1, 61):
        if seg < 10:
            seg_path = path + f"s0{seg}.txt"
        else:
            seg_path = path + f"s{seg}.txt"
        x = load_file(seg_path)  # (125, 45)
        X_sub.append(x)
    X_sub = np.stack(X_sub)  # (60, 125, 45)

    return X_sub


def load_act(path, subjects):
    X_act = []
    Sub = []

    for sub in subjects:
        sub_path = path + f"p{sub}/"
        X_sub = load_sub(sub_path)
        X_act.append(X_sub)  # (n_sub * 60, 125, 45)
        Sub.append(np.ones(X_sub.shape[0]) * sub)  # (n_sub * 60, 125, 45)

    X_act = np.concatenate(X_act)
    Sub = np.concatenate(Sub)

    return X_act, Sub


def load_dataset(subjects):
    prefix = "data/raw/DailySports/data/"

    data = []
    states = []
    sub_labels = []

    for act in range(1, 20):  # 19 activities
        if act == 16:  # Discard a similar activity, so that we can have 18 classes
            continue

        if act < 10:
            act_path = prefix + f"a0{act}/"
        else:
            act_path = prefix + f"a{act}/"

        x, sub = load_act(act_path, subjects)
        y = np.zeros(x.shape[0])

        if act < 16:
            y.fill(act)
        else:
            y.fill(act - 1)  # Revise the labels for cls 17 to 19 being 16 to 18

        data.append(x)
        states.append(y)
        sub_labels.append(sub)

    data = np.concatenate(data)  # (19 * n_sub * 60, 125, 45)
    states = np.concatenate(states)  # (19 * n_sub * 60, )
    sub_labels = np.concatenate(sub_labels)  # (19 * n_sub * 60, )

    return data, states, sub_labels


if __name__ == "__main__":

    # ######################## Download & Unzip ###########################
    raw = "data/raw/DailySports"

    if not os.path.exists(raw):
        os.makedirs(raw)

    wget.download(
        "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip",
        f"{raw}/DailySports.zip",
    )

    with ZipFile(f"{raw}/DailySports.zip", "r") as zObject:
        zObject.extractall(path=f"{raw}")
    # ######################## Download & Unzip ###########################

    trainX, trainy, train_sub_label = load_dataset([1, 2, 3, 4, 5, 6, 7, 8])
    trainX, testX, trainy, testy, train_sub_label, test_sub_label = train_test_split(
        trainX, trainy, train_sub_label, test_size=0.25, random_state=0, stratify=trainy
    )

    print(trainX.shape, trainy.shape, train_sub_label.shape)
    print(testX.shape, testy.shape, test_sub_label.shape)

    path = "data/saved/DailySports/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "x_train.pkl", "wb") as f:
        pickle.dump(trainX, f)
    with open(path + "x_test.pkl", "wb") as f:
        pickle.dump(testX, f)

    with open(path + "state_train.pkl", "wb") as f:
        pickle.dump(trainy - 1, f)
    with open(path + "state_test.pkl", "wb") as f:
        pickle.dump(testy - 1, f)

    with open(path + "subject_label_train.pkl", "wb") as f:
        pickle.dump(train_sub_label - 1, f)
    with open(path + "subject_label_test.pkl", "wb") as f:
        pickle.dump(test_sub_label - 1, f)
