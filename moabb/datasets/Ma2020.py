from moabb.datasets.base import BaseDataset
from moabb.datasets import download as dl

from scipy.io import loadmat
import numpy as np
import os
from mne.datasets.utils import _get_path, _do_path_update
from mne import annotations_from_events
from mne.io import read_raw_cnt
from mne.channels import make_standard_montage

instructions = """"""


class Ma2020(BaseDataset):
    """
    Same limb motor imagery dataset from Ma et al 2020.

    Different joint same limb motor imagery dataset from the paper [1]_.

    **Dataset description**

    This dataset contains 25 subjects with 19 runs containing 300 trials
    of each class (right hand, right elbow, and rest). The data was recorded
    at 1000Hz with 63 EEG channels.

    The subjects sat in a comfortable chair with their hands naturally on
    their thighs, keeping their eyes one-meter away from the screen. Each
    trial (8s) started with a white circle at the center of the monitor for
    2s, followed by a red circle as a cue for 1s to remind the subjects of
    paying attention to the upcoming target. The target prompt (“Hand” or
    “Elbow”) appeared on the screen for 4s. During this period, the subjects
    were asked to imagine the prompted movement kinesthetically in mind rather
    than a visual type of imagery. The subjects were instructed to avoid any
    motion during imagination. The EMG of the right hand and the right forearm
    of the subjects were monitored to make sure they did not move involuntarily.
    After the imagination, “Break” appeared for 1s as the end of the entire 8s
    trial. During the break, the subjects were asked to relax and minimize their
    eye and muscle movements.

    References
    ----------
    .. [1] Xuelin Ma, Shuang Qiu, and Huiguang He.  Multi-channel EEG recording
           during  motor  imagery  of  different  joints  from  the  same  limb.
           ScientificData, 7(1):1–9, 2020
    """
    def __init__(self):
        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=1,
            events=dict(right_hand=1, right_elbow=2, rest=3),
            code='Ma2020',
            interval=[0, 4],
            paradigm='imagery',
            doi='10.1038/s41597-020-0535-2')

    def _get_single_subject_data(self, subject):
        """return data for a single subejct"""

        file_paths = self.data_path(subject)
        sessions = {}
        sessions['session_1'] = {}
        for run in range(1, 20):
            raw = read_raw_cnt(file_paths[run - 1], preload=True)
            # Drop EMG channels
            remove_chs = ['HEO', 'VEO', 'EMG1', 'EMG2']
            emg_chs = [emg for emg in remove_chs if emg in raw.ch_names]
            raw = raw.drop_channels(emg_chs)
            # Modify the annotations to reflect actual classes + adding rest class
            stim = raw.annotations.description.astype(np.dtype('<U11'))
            if len(stim):
                stim[stim == '1'] = 'right_hand'
                stim[stim == '2'] = 'right_elbow'
                raw.annotations.description = stim
            else:
                sfreq = raw.info['sfreq']
                events = np.zeros((75, 3))
                events[:, 0] = np.arange(75.0)*sfreq*4.25
                events[:, -1] = 3
                raw = raw.set_annotations(annotations_from_events(events, sfreq, event_desc={3: 'rest'}))
            run_name = 'run_{}'.format(run)
            sessions['session_1'][run_name] = raw
        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        key = "MNE-ma2020-data"
        path = _get_path(path, key, "Ma2020")
        _do_path_update(path, True, key, "Ma2020")
        basepath = os.path.join(path, key)
        if not os.path.isdir(basepath):
            print(instructions)
            raise FileNotFoundError

        subject_paths = []
        for session in range(1, 20):
            if 1 <= session < 16:
                session_path = "sourcedata\sub-{0:03d}\ses-{1:02d}\eeg\sub-{0:03d}_ses-{1:02d}_task-motorimagery_eeg.cnt".format(subject, session)
            else:
                session_path = "sourcedata\sub-{0:03d}\ses-{1:02d}\eeg\sub-{0:03d}_ses-{1:02d}_task-rest_eeg.cnt".format(subject, session)

            subject_paths.append(os.path.join(basepath, session_path))
        return subject_paths
