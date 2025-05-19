import gdown
from zipfile import ZipFile


def download_data():
    file_urls = [
        "1bgs6YwpN4EgL5X-fbtQ06lsVbkHaer8r",
        "1w0mDXTOy_Xj4XBHFn_GyKQCb0frdG5Wr",
        "146JrCv7NqW1s3818Tbuq8hTEcqviXlqY",
        "1x6s83JKkYqeWeV7M8GAI-Isn8C9YeSZE",
        "1GVy7JQDfCUlk-aI-E5I7HVCjQKFaCdm3",
    ]
    file_outpus = [
        "DailySports.zip",
        "GRABMyo.zip",
        "HAR_inertial.zip",
        "UWave.zip",
        "WISDM.zip",
    ]

    for idx in range(5):
        url = f"https://drive.google.com/uc?id={file_urls[idx]}"
        output = f"data/saved/{file_outpus[idx]}"
        gdown.download(url, output)


def unzip_data():
    files = [
        "DailySports.zip",
        "GRABMyo.zip",
        "HAR_inertial.zip",
        "UWave.zip",
        "WISDM.zip",
    ]

    for file in files:
        with ZipFile(f"data/saved/{file}", "r") as zObject:
            zObject.extractall(path=f"data/saved")
