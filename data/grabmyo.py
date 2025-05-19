import wget
import numpy as np
import scipy.io as scio
import os
import pickle
from tsai.data.preparation import SlidingWindow
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from zipfile import ZipFile


physionet_root = "data/raw/GRABMyo/mat_file/"
data_dir = physionet_root + "Session1_converted/"

DOWNSAMPLE = True
resample_length = 256 * 5  # downsample to 256 hz
window_len_grabmyo = 128  # each window is 1 sec
stride = 128  # no overlapping

GROUP = "combined"
if GROUP == "forearm":
    input_channels_grabmyo = 16
elif GROUP == "wrist":
    input_channels_grabmyo = 12
elif GROUP == "combined":
    input_channels_grabmyo = 28

N_sessions = 1
N_subjects = 43
N_classes = 16
N_trails = 7
# N_train_trails = 5


###################### from wfdb to mat ######################
def convert_grabmyo():
    # https://physionet.org/content/grabmyo/1.1.0/grabmyo_convert_wfdb_to_mat.py
    """
    This script will read the .dat and .hea files downloaded locally to
    to your hard drive. The downloaded files are organzied in 3 folders
    'Session 1','Session 2' and 'Session 3'. Each folder contains 119 data
    files (.dat) and 119 header files (.hea).

    signal properties
    total channels = 28 (16 forearm + 12 wrist)
    sampling frequency = 2048 Hz
    bandpass filtering (hardware) = 10Hz-500Hz

    In order to run this script make sure the above three folders are and
    a fileconversion function 'rdwfdb.m' are in the same directory

    output %%%%%%%%%
    Main Folder: 'Output BM'
    Folders: 'Session 1_converted','Session 2_converted', 'Session 3_converted',
    Each folder: 43 .mat files
    VarOut: DATA_FOREARM, DATA_WRIST (7x17 cell matrices)
    DATA_FOREARM: each cell: 5secs*sampfreq x Nchannels numeric array
    DATA_WRIST: each cell: 5secs*sampfreq x Nchannels numeric array

    Forearm Electrode Configuration %%%%%%%%%
    1  2  3  4  5  6  7  8
    9 10 11 12 13 14 15 16

    Wrist Electrode Configuration %%%%%%%%%
    1  2  3  4  5  6
    7  8  9 10 11 12

    Written by Ashirbad Pradhan
    email: ashirbad.pradhan@uwaterloo.ca
    """

    # Your Python code starts here

    import os
    import sys
    import wfdb
    import shutil
    import numpy as np
    from scipy.io import savemat

    # Add paths for Session1, Session2, and Session3
    path = "data/raw/GRABMyo/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.2"
    session_paths = ["Session1"]  # or ["Session1", "Session2", "Session3"]
    for session_path in session_paths:
        sys.path.append(os.path.join(os.getcwd(), path, session_path))

    # Obtain the total number of subjects
    nsub = (
        # len(os.listdir(os.path.join(os.getcwd(), path, "Session1"))) - 1  <=> no need to subtract by 1
        len(os.listdir(os.path.join(os.getcwd(), path, "Session1")))
    )  # Assuming number of subjects are the same in all sessions
    nsession = 1
    ngesture = 16  # Total number of gestures
    ntrials = 7  # Total number of trials

    # Define output folder
    output_folder = "data/raw/GRABMyo/mat_file"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    else:
        while True:
            print(f"Found existing folder in: {os.getcwd()}")
            # cont = input("Overwrite it (Y/N)? ").upper()
            cont = "Y"
            if cont in ("Y", "N"):
                if cont == "Y":
                    print("Overwriting")
                    shutil.rmtree(output_folder)
                    os.mkdir(output_folder)
                    break
                else:
                    print("Exiting Script!")
                    sys.exit()

    foldername = []
    filename = []
    flag = 0
    count = 0

    # Define channel mappings for forearm and wrist
    forearm_channels = np.concatenate(
        (np.ones(8), np.ones(8), np.zeros(8), np.zeros(8))
    )
    wrist_channels = np.concatenate(
        (
            np.zeros(8),
            np.zeros(8),
            np.zeros(1),
            np.ones(6),
            np.zeros(2),
            np.ones(6),
            np.zeros(1),
        )
    )
    indices = [i for i, x in enumerate(wrist_channels) if x == 1]
    print(indices)

    # Define data_forearm and data_wrist lists before the loop

    # Create a 7x17 array of 2D matrices
    matrices_forearm = np.empty((7, 17), dtype=object)
    matrices_wrist = np.empty((7, 17), dtype=object)

    # Populate each element with a 2D matrix (for demonstration, using zero data)
    for i in range(7):
        for j in range(17):
            matrices_forearm[i, j] = np.zeros((10240, 16), dtype=np.float64)
            matrices_wrist[i, j] = np.zeros((10240, 12), dtype=np.float64)

    foldername = []
    filename = []
    flag = 0
    count = 0

    for isession in range(1, nsession + 1):  # Total number of sessions per participant
        converted_folder = f"Session{isession}_converted"
        os.makedirs(
            os.path.join(output_folder, converted_folder),
            exist_ok=True,
        )

        for isub in range(1, nsub + 1):
            foldername = f"session{isession}_participant{isub}"

            for igesture in range(1, ngesture + 2):  # +1 to include rest gesture

                for itrial in range(1, ntrials + 1):
                    filename = f"session{isession}_participant{isub}_gesture{igesture}_trial{itrial}"
                    filepath = os.path.join(
                        os.getcwd(), path, f"Session{isession}", foldername, filename
                    )

                    # Load WFDB data
                    record = wfdb.rdrecord(filepath)

                    # Extract signals and other information
                    data_emg = record.p_signal
                    fs = record.fs

                    # Extract forearm and wrist data based on channel mappings
                    data_forearm = data_emg[:, forearm_channels.astype(bool)]
                    data_wrist = data_emg[:, wrist_channels.astype(bool)]

                    # Assuming DATA_FOREARM and DATA_WRIST are lists
                    matrices_forearm[itrial - 1, igesture - 1] = data_forearm
                    matrices_wrist[itrial - 1, igesture - 1] = data_wrist

            count += 1
            print(f"Converted: {count} of {nsub * nsession} files")

            # Create a dictionary to hold the data
            savemat(
                os.path.join(
                    output_folder,
                    converted_folder,
                    f"{foldername}.mat",
                ),
                {"DATA_FOREARM": matrices_forearm, "DATA_WRIST": matrices_wrist},
            )


def apply_sliding_window(record, cls, window_len, stride):
    sliding_windows, _ = SlidingWindow(
        window_len=window_len,
        stride=stride,
        seq_first=True,
        pad_remainder=True,
        padding_value=0,
        add_padding_feature=False,
    )(record)

    sliding_windows = np.transpose(sliding_windows, (0, 2, 1))
    windows_labels = np.ones(sliding_windows.shape[0], dtype=float) * cls

    return sliding_windows, windows_labels


def extract_samples_with_sliding_windows(
    window_len, stride, group="forearm", resampling=True
):
    train_data, test_data = list(), list()
    train_labels, test_labels = list(), list()

    for session in range(1, N_sessions + 1):
        for sub in range(1, N_subjects + 1):
            filepath = data_dir + "session{}_participant{}.mat".format(session, sub)
            mat = scio.loadmat(filepath)  # Dictionary

            if group == "forearm":
                collections = mat["DATA_FOREARM"]  #  7 * 17, trials * activities
            elif group == "wrist":
                collections = mat["DATA_WRIST"]  # 7 * 17,
            elif group == "combined":
                collections_forearm = mat["DATA_FOREARM"]
                collections_wrist = mat["DATA_WRIST"]
            else:
                raise ValueError("Wrong sensor group is given")

            ###################### Train-Test split on all trials ######################
            for trail in range(0, N_trails):
                for cls in range(0, N_classes):  # Discard the last class 'rest'
                    if group in ["forearm", "wrist"]:
                        record = collections[trail][
                            cls
                        ]  # 10240 time steps, 5 sec of 2048 hz
                        record = np.nan_to_num(record)
                    else:
                        record = np.concatenate(
                            (
                                collections_forearm[trail][cls],
                                collections_wrist[trail][cls],
                            ),
                            axis=1,
                        )
                        record = np.nan_to_num(record)

                    if resampling:
                        record = resample(record, resample_length)

                    sliding_windows, windows_labels = apply_sliding_window(
                        record, cls, window_len, stride
                    )

                    train_data.append(sliding_windows)
                    train_labels.append(windows_labels)

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels)

    return train_data, train_labels


if __name__ == "__main__":

    # ######################## Download & Unzip ###########################
    # raw = "data/raw/GRABMyo"

    # if not os.path.exists(raw):
    #     os.makedirs(raw)

    # wget.download(
    #     "https://physionet.org/static/published-projects/grabmyo/gesture-recognition-and-biometrics-electromyogram-grabmyo-1.0.2.zip",
    #     f"{raw}/GRABMyo.zip",
    # )

    # with ZipFile(f"{raw}/GRABMyo.zip", "r") as zObject:
    #     zObject.extractall(path=f"{raw}")
    # ######################## Download & Unzip ###########################

    convert_grabmyo()

    trainX, trainy = extract_samples_with_sliding_windows(
        window_len_grabmyo, stride, group=GROUP, resampling=DOWNSAMPLE
    )
    trainX, testX, trainy, testy = train_test_split(
        trainX, trainy, test_size=0.25, random_state=0, stratify=trainy
    )

    print(trainX.shape, trainy.shape)
    print(testX.shape, testy.shape)

    ## Save signals to file
    path = "data/saved/GRABMyo/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/x_train.pkl", "wb") as f:
        pickle.dump(trainX, f)  # trainX tasks around 500 MB
    with open(path + "/x_test.pkl", "wb") as f:
        pickle.dump(testX, f)
    with open(path + "/state_train.pkl", "wb") as f:
        pickle.dump(trainy, f)
    with open(path + "/state_test.pkl", "wb") as f:
        pickle.dump(testy, f)
