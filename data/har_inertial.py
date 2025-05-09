import wget
import numpy as np
import os
import pickle
from pandas import read_csv
from numpy import dstack
from sklearn.model_selection import train_test_split
from zipfile import ZipFile


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, sep="\s+")
    return dataframe.values


# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=""):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)

    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)

    return loaded


def map_elements_to_sorted_indices(arr):
    # Extract unique elements and sort them
    unique_elements = np.unique(arr)
    sorted_elements = np.sort(unique_elements)

    # Create a dictionary mapping each element to its index in the sorted array
    element_to_index = {
        element: index for index, element in enumerate(sorted_elements, start=1)
    }

    # Map each element in the original array to its corresponding index
    mapped_arr = np.array([[element_to_index[element[0]]] for element in arr])

    return mapped_arr


# load a dataset group, such as train or test
def load_dataset(group, prefix=""):
    filepath = prefix + group + "/Inertial Signals/"

    # load all 9 files as a single array
    filenames = list()

    # total acceleration
    filenames += [
        "total_acc_x_" + group + ".txt",
        "total_acc_y_" + group + ".txt",
        "total_acc_z_" + group + ".txt",
    ]

    # body acceleration
    filenames += [
        "body_acc_x_" + group + ".txt",
        "body_acc_y_" + group + ".txt",
        "body_acc_z_" + group + ".txt",
    ]

    # body gyroscope
    filenames += [
        "body_gyro_x_" + group + ".txt",
        "body_gyro_y_" + group + ".txt",
        "body_gyro_z_" + group + ".txt",
    ]

    # load input data
    X = load_group(filenames, filepath)

    # load class outputs
    y = load_file(prefix + group + "/y_" + group + ".txt")

    # load subjects
    sub = load_file(prefix + group + f"/subject_{group}" + ".txt")
    sub = map_elements_to_sorted_indices(sub)

    return X, y, sub


if __name__ == "__main__":

    # ######################## Download & Unzip ###########################
    raw = "data/raw/HAR_inertial"

    if not os.path.exists(raw):
        os.makedirs(raw)

    wget.download(
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        f"{raw}/HAR_inertial.zip",
    )

    with ZipFile(f"{raw}/HAR_inertial.zip", "r") as zObject:
        zObject.extractall(path=f"{raw}")

    with ZipFile(f"{raw}/UCI HAR Dataset.zip", "r") as zObject:
        zObject.extractall(path=f"{raw}")
    # ######################## Download & Unzip ###########################

    prefix = "data/raw/HAR_inertial/UCI HAR Dataset/"

    # load all train
    trainX, trainy, train_sub_label = load_dataset("train", prefix)
    trainX, valX, trainy, valy, train_sub_label, val_sub_label = train_test_split(
        trainX, trainy, train_sub_label, test_size=0.1, random_state=0
    )
    print(trainX.shape, trainy.shape, train_sub_label.shape)
    print(valX.shape, valy.shape, val_sub_label.shape)

    # load all test
    testX, testy, test_sub_label = load_dataset("test", prefix)
    print(testX.shape, testy.shape, test_sub_label.shape)

    # save signals to file
    path = "data/saved/HAR_inertial/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/x_train.pkl", "wb") as f:
        pickle.dump(trainX, f)
    with open(path + "/x_val.pkl", "wb") as f:
        pickle.dump(valX, f)
    with open(path + "/x_test.pkl", "wb") as f:
        pickle.dump(testX, f)

    with open(path + "/state_train.pkl", "wb") as f:
        pickle.dump(trainy - 1, f)
    with open(path + "/state_val.pkl", "wb") as f:
        pickle.dump(valy - 1, f)
    with open(path + "/state_test.pkl", "wb") as f:
        pickle.dump(testy - 1, f)

    with open(path + "/subject_label_train.pkl", "wb") as f:
        pickle.dump(train_sub_label - 1, f)
    with open(path + "/subject_label_val.pkl", "wb") as f:
        pickle.dump(val_sub_label - 1, f)
    with open(path + "/subject_label_test.pkl", "wb") as f:
        pickle.dump(test_sub_label - 1, f)
