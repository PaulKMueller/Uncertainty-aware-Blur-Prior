import argparse
import cProfile
import os
import pstats
from datetime import datetime

import mne
import numpy as np
import scipy
import torch
from sklearn.discriminant_analysis import _cov
from sklearn.utils import shuffle
from tqdm import tqdm

# Set mne log level to warning to reduce output
mne.set_log_level("WARNING")


def get_args_parser():
    parser = argparse.ArgumentParser("train", add_help=False)
    parser.add_argument("--subject", type=int)
    return parser.parse_args()


args = get_args_parser()

sub = args.subject


NUMBER_OF_SESSIONS = 4
PROFILING_DIR = "profiling"
os.makedirs(PROFILING_DIR, exist_ok=True)

SEED = 20200220
RESAMPLING_FREQUENCY = 250
tmin = -0.2
tmax = 1.0
WHITEN = True

PROJECT_DIR = "data/things-eeg"

if WHITEN:
    save_dir = os.path.join(
        PROJECT_DIR,
        f"Preprocessed_data_{RESAMPLING_FREQUENCY}Hz_whiten",
        "sub-" + format(sub, "02"),
    )
else:
    save_dir = os.path.join(
        PROJECT_DIR,
        f"Preprocessed_data_{RESAMPLING_FREQUENCY}Hz_no_whiten",
        "sub-" + format(sub, "02"),
    )
os.makedirs(save_dir, exist_ok=True)

CHANNEL_ORDER = [ "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", ]  # fmt: skip
MVNN_DIM = "epochs"
STIM_CHANNEL = "stim"


def mvnn(epoched_test, epoched_train):
    whitened_test = []
    whitened_train = []
    for s in tqdm(range(NUMBER_OF_SESSIONS), desc="MVNN whitening sessions"):
        session_data = [epoched_test[s], epoched_train[s]]

        # Data partitions covariance matrix of shape:
        # Data partitions × EEG channels × EEG channels
        sigma_part = np.empty(
            (len(session_data), session_data[0].shape[2], session_data[0].shape[2])
        )
        for p in range(sigma_part.shape[0]):
            # Image conditions covariance matrix of shape:
            # Image conditions × EEG channels × EEG channels
            sigma_cond = np.empty(
                (
                    session_data[p].shape[0],
                    session_data[0].shape[2],
                    session_data[0].shape[2],
                )
            )
            for i in tqdm(range(session_data[p].shape[0])):
                cond_data = session_data[p][i]
                # Compute covariace matrices at each time point, and then
                # average across time points
                if MVNN_DIM == "time":
                    sigma_cond[i] = np.mean(
                        [
                            _cov(cond_data[:, :, t], shrinkage="auto")
                            for t in range(cond_data.shape[2])
                        ],
                        axis=0,
                    )
                # Compute covariace matrices at each epoch (EEG repetition),
                # and then average across epochs/repetitions
                elif MVNN_DIM == "epochs":
                    sigma_cond[i] = np.mean(
                        [
                            _cov(np.transpose(cond_data[e]), shrinkage="auto")
                            for e in range(cond_data.shape[0])
                        ],
                        axis=0,
                    )
            # Average the covariance matrices across image conditions
            sigma_part[p] = sigma_cond.mean(axis=0)
        # # Average the covariance matrices across image partitions
        # sigma_tot = sigma_part.mean(axis=0)
        # ? It seems not fair to use test data for mvnn, so we change to just use training data
        sigma_tot = sigma_part[1]
        # Compute the inverse of the covariance matrix
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

        ### Whiten the data ###
        whitened_test.append(
            np.reshape(
                (
                    np.reshape(
                        session_data[0],
                        (-1, session_data[0].shape[2], session_data[0].shape[3]),
                    ).swapaxes(1, 2)
                    @ sigma_inv
                ).swapaxes(1, 2),
                session_data[0].shape,
            )
        )
        whitened_train.append(
            np.reshape(
                (
                    np.reshape(
                        session_data[1],
                        (-1, session_data[1].shape[2], session_data[1].shape[3]),
                    ).swapaxes(1, 2)
                    @ sigma_inv
                ).swapaxes(1, 2),
                session_data[1].shape,
            )
        )

    return whitened_test, whitened_train


def epoch_data(mode, sub, use_decim=True):
    epoched_data = []
    img_conditions = []

    for s in tqdm(
        range(NUMBER_OF_SESSIONS), desc=f"Epoching {mode} data for subject {sub}"
    ):
        ### Load the EEG data ###
        eeg_dir = os.path.join(
            "Raw_data",
            "sub-" + format(sub, "02"),
            "ses-" + format(s + 1, "02"),
            f"raw_eeg_{mode}.npy",
        )
        eeg_data = np.load(os.path.join(PROJECT_DIR, eeg_dir), allow_pickle=True).item()

        info = mne.create_info(
            eeg_data["ch_names"], eeg_data["sfreq"], eeg_data["ch_types"]
        )
        ch_names = info["ch_names"]
        raw = mne.io.RawArray(eeg_data["raw_eeg_data"], info)
        sfreq = raw.info["sfreq"]

        events = mne.find_events(raw, stim_channel=STIM_CHANNEL, verbose=False)

        decim_factor = 1

        # Check if we want to use decimation and if the math works (must be integer)
        if use_decim and (sfreq % RESAMPLING_FREQUENCY == 0):
            decim_factor = int(sfreq / RESAMPLING_FREQUENCY)

            # We filter at Nyquist / 1.5 (re_sfreq / 3.0) to be safe
            raw.filter(
                l_freq=None, h_freq=RESAMPLING_FREQUENCY / 3.0, n_jobs=-1, verbose=False
            )

        elif RESAMPLING_FREQUENCY < sfreq:
            # Fallback to slow resampling if ratios don't match or use_decim is False
            print(f"Resampling raw data from {sfreq}Hz to {RESAMPLING_FREQUENCY}Hz...")
            raw, events = raw.resample(
                RESAMPLING_FREQUENCY, events=events, n_jobs=-1, stim_picks=STIM_CHANNEL
            )

        raw.pick_channels(CHANNEL_ORDER, ordered=True)
        idx_target = np.where(events[:, 2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

        epochs = mne.Epochs(
            raw,
            events,
            tmin=tmin,
            tmax=tmax,
            baseline=(None, 0),
            decim=decim_factor,  # <--- The speedup happens here
            preload=True,
            verbose=False,
        )

        epochs.crop(tmin=tmax - 1.0, tmax=tmax)

        data = epochs.get_data()
        times = epochs.times
        events = epochs.events[:, 2]
        img_cond = np.unique(events)

        if mode == "test":
            max_rep = 20
        else:
            max_rep = 2

        sorted_data = np.zeros((len(img_cond), max_rep, data.shape[1], data.shape[2]))

        for i in range(len(img_cond)):
            idx = np.where(events == img_cond[i])[0]
            idx = shuffle(idx, random_state=SEED, n_samples=max_rep)
            sorted_data[i] = data[idx]

        epoched_data.append(sorted_data)
        img_conditions.append(img_cond)

    return epoched_data, img_conditions, ch_names, times


with cProfile.Profile() as profiler:
    eeg_test, _, ch_names, times = epoch_data("test", sub)
    eeg_train, img_conditions_train, _, _ = epoch_data("training", sub)

    if WHITEN:
        whitened_test, whitened_train = mvnn(eeg_test, eeg_train)
        del eeg_test, eeg_train
    else:
        whitened_test = eeg_test
        whitened_train = eeg_train

    session_list = np.zeros((200, 80))
    for s in range(NUMBER_OF_SESSIONS):
        if s == 0:
            merged_test = whitened_test[s]
        else:
            merged_test = np.append(merged_test, whitened_test[s], 1)
        start_index = merged_test.shape[1] - whitened_test[s].shape[1]
        end_index = merged_test.shape[1]
        session_list[:, start_index:end_index] = s

    del whitened_test

    # 'img': duplicated_images,
    # 'label': label,
    img_directory = "data/things-eeg/Image_set_Resize/test_images"
    all_folders = [
        d
        for d in os.listdir(img_directory)
        if os.path.isdir(os.path.join(img_directory, d))
    ]
    all_folders.sort()
    images = []
    labels = []
    texts = []
    for i, folder in enumerate(all_folders):
        folder_path = os.path.join(img_directory, folder)
        all_images = [
            img
            for img in os.listdir(folder_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        all_images.sort()
        images.extend(
            os.path.join(folder_path, img).rsplit("Image_set_Resize/")[-1]
            for img in all_images
        )
        labels.extend([i for img in all_images])
        texts.extend([img.rsplit("_", 1)[0] for img in all_images])
    img_list = np.tile(np.array(images)[:, np.newaxis], (1, 80))
    labels_list = np.tile(np.array(labels)[:, np.newaxis], (1, 80))
    text_list = np.tile(np.array(texts)[:, np.newaxis], (1, 80))

    test_dict = {
        "eeg": merged_test.astype(np.float16),
        "label": labels_list,
        "img": img_list,
        "text": text_list,
        "session": session_list,
        "ch_names": ch_names,
        "times": times,
    }

    print(f"Saving test data to {save_dir}...")
    torch.save(test_dict, os.path.join(save_dir, "test.pt"), pickle_protocol=5)

    ### Merge and save the training data ###
    ses_list = np.zeros((33080, 2))
    for s in range(NUMBER_OF_SESSIONS):
        if s == 0:
            white_data = whitened_train[s]
            img_cond = img_conditions_train[s]
        else:
            white_data = np.append(white_data, whitened_train[s], 0)
            img_cond = np.append(img_cond, img_conditions_train[s], 0)
        start_index = white_data.shape[0] - whitened_train[s].shape[0]
        end_index = white_data.shape[0]
        ses_list[start_index:end_index] = s

    del whitened_train
    # Data matrix of shape:
    # Image conditions × EGG repetitions × EEG channels × EEG time points
    merged_train = np.zeros(
        (
            len(np.unique(img_cond)),
            white_data.shape[1] * 2,
            white_data.shape[2],
            white_data.shape[3],
        )
    )

    sorted_session_list = np.zeros((16540, 4))
    for i in range(len(np.unique(img_cond))):
        # Find the indices of the selected category
        idx = np.where(img_cond == i + 1)[0]

        for r in range(len(idx)):
            sorted_session_list[i][r * 2 : r * 2 + 2] = ses_list[idx[r]]
            if r == 0:
                ordered_data = white_data[idx[r]]
            else:
                ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
        merged_train[i] = ordered_data

    del ordered_data

    img_directory = "data/things-eeg/Image_set_Resize/training_images"
    all_folders = [
        d
        for d in os.listdir(img_directory)
        if os.path.isdir(os.path.join(img_directory, d))
    ]
    all_folders.sort()
    images = []
    labels = []
    texts = []
    for (
        i,
        folder,
    ) in enumerate(  # `all_folders` is a list that is being used to store the names of all the
        # folders present in the specified directory `img_directory`. In this code
        # snippet, `all_folders` is being populated with the names of the folders
        # found within the `img_directory` path. These folders are then sorted in
        # alphabetical order.
        all_folders
    ):
        folder_path = os.path.join(img_directory, folder)
        all_images = [
            img
            for img in os.listdir(folder_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        all_images.sort()
        images.extend(
            os.path.join(folder_path, img).rsplit("Image_set_Resize/")[-1]
            for img in all_images
        )
        labels.extend([i for img in all_images])
        texts.extend([img.rsplit("_", 1)[0] for img in all_images])

    labels_list = np.tile(np.array(labels)[:, np.newaxis], (1, 4))
    img_list = np.tile(np.array(images)[:, np.newaxis], (1, 4))
    text_list = np.tile(np.array(texts)[:, np.newaxis], (1, 4))

    train_dict = {
        "eeg": merged_train.astype(np.float16),
        "label": labels_list,
        "img": img_list,
        "text": text_list,
        "session": sorted_session_list,
        "ch_names": ch_names,
        "times": times,
    }
    # Create the directory if not existing and save the data
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving training data to {save_dir}...")
    file_name_train = "train.pt"
    torch.save(train_dict, os.path.join(save_dir, file_name_train), pickle_protocol=5)

stats = pstats.Stats(profiler)
stats.sort_stats(pstats.SortKey.CUMULATIVE)
stats.dump_stats(
    os.path.join(
        PROFILING_DIR, f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
    )
)
