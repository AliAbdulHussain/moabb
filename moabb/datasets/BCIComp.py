from moabb.datasets.base import BaseDataset

# download related libs
from moabb.datasets import download as dl
import requests
from mne.utils.progressbar import ProgressBar
import time
import shutil
import zipfile as z
import os
import os.path as op

# Data manipulation libs
from scipy.io import loadmat
from mne import create_info
from mne.channels import make_standard_montage
import numpy as np
from mne.io import RawArray

import logging

logger = logging.getLogger()

trainURL = "http://bbci.de/competition/download/competition_iii/berlin/1000Hz/"
testURL = "https://www.bbci.de/competition/iii/results/"


def download_train(subject: str, sign: str, timeout=30) -> str:
    url = "{}data_set_IVa_{}_mat.zip".format(trainURL, subject)
    base_path, key, path = dl.create_destination(trainURL, sign)
    zippath = op.join(base_path, "data_set_IVa_{}_mat.zip".format(subject))
    matpath = op.join(base_path, op.join("1000Hz", "data_set_IVa_{}.mat").format(subject))
    if not op.isfile(matpath):
        if not op.isfile(zippath):
            auth = ('amar.yousif09@gmail.com', 'AThaLe9ied')
            response = requests.get(url, auth=auth, timeout=timeout, stream=True)
            zippath = download_file(zippath, response)
        with z.ZipFile(zippath, 'r') as f:
            f.extractall(base_path)
        os.remove(zippath)
    return matpath


def download_test(subject: str, sign: str, timeout=30) -> str:
    url = "{}berlin_IVa/true_labels_{}.mat".format(testURL, subject)
    destination, key, path = dl.create_destination(url, sign)
    if not op.isfile(destination):
        response = requests.get(url, timeout=timeout, stream=True, verify=False)
        download_file(destination, response)
    return destination


def download_file(destination: str, response) -> str:
    """
    Downloads the response file into the given destination
    :param destination: the destination as which to downlad the file to
    :param response: HTTP/s response corresponding to the download file
    :return: the destination as which the file was downlaoded to
    """
    temp_file_name = destination + ".part"
    initial_size = 0
    file_size = int(response.headers.get('Content-Length', '0').strip())
    file_size += initial_size
    mode = 'ab' if initial_size > 0 else 'wb'
    progress = ProgressBar(file_size, initial_size, unit='B',
                           mesg='Downloading', unit_scale=True,
                           unit_divisor=1024)
    del file_size
    chunk_size = 8192  # 2 ** 13
    with open(temp_file_name, mode) as local_file:
        t0 = time.time()
        for chunk in response.iter_content(chunk_size):
            dt = time.time() - t0
            if dt < 0.01:
                chunk_size *= 2
            elif dt > 0.1 and chunk_size > 8192:
                chunk_size = chunk_size // 2
            if not chunk:
                break
            local_file.write(chunk)
            progress.update_with_increment_value(len(chunk))
            t0 = time.time()
    shutil.move(temp_file_name, destination)
    return destination


class Dataset4a(BaseDataset):
    """
    Motor Imagery dataset4a from competition 3.

    BCI competition 3 dataset 4a [1]_.

    **Dataset Description**

    The dataset comprises of EEG recording from 5 subjects. These subjects were
    asked to perform right hand and right foot imaginations for 3.5s with random
    rest periods of 1.75s to 2.25s. These were indicated either by using letters
    appearing behind a fixation cross (which might nevertheless induce little
    target-correlated eye movements) or a randomly moving object indicated targets
    (inducing target-uncorrelated eye movements). Each subject performed 280 cues
    (140 of each class). Each subject performed 280 cues. The data was recorded
    using BrainAmp amplifiers and 128 channel Ag/AgCI cap (uses an extended
    international 10/20-system) from ECI with 118 EEG electrodes.


    References
    ----------

    .. [1] Benjamin  Blankertz,  Klaus  Robert  M̈uller,  Dean  J.  Krusienski,
           GerwinSchalk, Jonathan R. Wolpaw, Alois Schl̈ogl, Gert Pfurtscheller,
           Jos ́e Del R.Milĺan, Michael Schr̈oder, and Niels Birbaumer.  The
           BCI competition III:Validating alternative approaches to actual BCI
           problems. IEEE  Trans-actions on Neural Systems and Rehabilitation
           Engineering, 14(2):153–159,2006.
    """

    def __init__(self):
        super(Dataset4a, self).__init__(subjects=list(range(1, 6)),
                                        sessions_per_subject=4,
                                        events={'right_hand': 1, 'right_foot': 2},
                                        code='Comp3Dataset4a',
                                        interval=[0, 3.5],
                                        paradigm='imagery',
                                        doi='10.1109/TBME.2004.827088')
        self.subject_names = ['aa', 'al', 'av', 'aw', 'ay']

    def _get_single_subject_data(self, subject):
        [train_path, test_path] = self.data_path(subject)
        train_data = loadmat(train_path)
        test_data = loadmat(test_path)

        # Create channel info and montage
        eeg_ch_names = train_data['nfo'][0, 0][2][0]
        ch_names = [elem[0] for elem in eeg_ch_names] + ['stim']
        ch_types = ['eeg'] * 118 + ['stim']
        sfreq = train_data['nfo'][0, 0][1][0, 0]
        info = create_info(ch_names=ch_names,
                           ch_types=ch_types,
                           sfreq=sfreq)
        montage = make_standard_montage('standard_1005')

        # Create raw_data
        raw_data = train_data['cnt'].T

        # Create raw_event
        raw_event = np.zeros((1, raw_data.shape[1]))
        cue_pos = train_data['mrk'][0, 0][0]
        sep_idx = np.min(test_data['test_idx'])
        sep_pos = cue_pos[0, sep_idx]
        raw_event[:, cue_pos] = test_data['true_y']
        data = np.concatenate([raw_data, raw_event], axis=0)

        # Create RawArray
        raw_train = RawArray(data=data[:, :sep_pos], info=info, verbose=False)
        raw_test = RawArray(data=data[:, sep_pos:], info=info, verbose=False)
        # FIXME: some unknown channels in the dataset ['FAF5',
        #  'FAF1', 'FAF2', 'FAF6', 'FFC7', 'FFC8', 'CFC7', 'CFC5',
        #  'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'CCP7',
        #  'CCP8', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4',
        #  'PCP6', 'PCP8', 'OPO1', 'OPO2']
        # raw.set_montage(montage)

        return {'session_1': {'train': raw_train, 'test': raw_test}}

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        """Download the data from one subject"""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject.")

        subject = self.subject_names[subject - 1]
        train_path = download_train(subject, 'BCIComp')
        test_path = download_test(subject, 'BCIComp')
        return [train_path, test_path]
