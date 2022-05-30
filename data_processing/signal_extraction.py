import os

import librosa
import numpy as np
import scipy.io as sio
import scipy.signal
import speechpy
import re

from scipy.io import wavfile


class DataExtractor:
    @staticmethod
    def read_physionet_mat(file_path):
        mat = sio.loadmat(file_path)  # load mat-file
        mdata = mat['example_data']  # variable in mat file
        ndata = {n: mdata[n][0, 0] for n in mdata.dtype.names}
        pcg_recordings = ndata['example_audio_data'].squeeze()
        patient_ids = ndata['patient_number'].squeeze()
        return pcg_recordings, patient_ids

    @staticmethod
    def extract_circor_labels(file_path, sampling_rate, sound, extension='txt'):
        """
        Parameters
        ----------
        file_path: the path of the observation
        sampling_rate: the sampling rate used to collect the sound
        sound: the output of the wav file of the observation

        Returns
        -------
        A numpy ndarray with the labels in the frequency domain
        """
        sound_duration = len(sound) / sampling_rate
        time_indices = np.arange(start=0., stop=sound_duration, step=1 / sampling_rate)

        f = np.genfromtxt(f'{file_path}.{extension}', delimiter='\t')  # 0: start; 1: end; 2: state
        labels_time = f[:, 2].astype(np.int32)
        labels_fs = np.zeros((len(time_indices)), dtype=np.int32)
        j = 0
        for i in range(len(time_indices)):
            if time_indices[i] < f[j, 1]:
                labels_fs[i] = labels_time[j]
            else:
                try:
                    j = j + 1
                    labels_fs[i] = labels_time[j]
                except:
                    labels_fs[i:] = labels_time[j - 1]
                    return labels_fs

        return labels_fs

    @staticmethod
    def get_annotated_intervals(labels, name):
        annotated_indices = np.where(labels > 0)[0]
        if len(annotated_indices) == 0:
            return None
        annotated_intervals = []
        start = end = annotated_indices[0]
        suffix = 0
        for i in range(1, len(annotated_indices)):
            if (annotated_indices[i] - end) == 1:
                end = annotated_indices[i]
            else:
                annotated_intervals.append((start, end, f'{name}_{suffix}'))
                start = end = annotated_indices[i]
                suffix += 1
        if len(annotated_intervals) == 0:  # Condition when there are no breaks in annotation
            annotated_intervals.append((start, end, f'{name}_{suffix}'))
        if (start, end, f'{name}_{suffix}') != annotated_intervals[-1]:  # Condition for last element
            annotated_intervals.append((start, end, f'{name}_{suffix}'))
        return annotated_intervals

    @staticmethod
    def get_circor_noisy_labels(label_):
        label_ = np.copy(label_)
        label_ -= 1
        wrong_labels = []
        for t in range(1, len(label_)):
            if (label_[t] != -1) and label_[t - 1] != -1:
                if (label_[t] != label_[t - 1]) and (label_[t] != (label_[t - 1] + 1) % 4):
                    wrong_labels.append(t)
        return np.array(wrong_labels)

    @staticmethod
    def read_circor_raw(dataset_path, extension='txt'):
        """
        Parameters
        ----------
        extension: the extension of the label files
        dataset_path: the directory containing the raw circor dataset train data

        Returns
        -------
            A numpy np.ndarray containing N obersvations, with features "id", "sound" and "label"
        """
        if not dataset_path.endswith('/'):
            print("Error. Please provide directory path.")
            return None
        recordings = sorted([f for f in os.listdir(dataset_path) if f.endswith('.wav')])
        dataset = np.zeros([len(recordings), 3], dtype=np.object)
        i = skipped = 0
        for recording in recordings:
            name = re.split('\.', recording)[0]
            sampling_rate, sound = wavfile.read(f"{dataset_path}{name}.wav")
            try:
                labels = DataExtractor.extract_circor_labels(f"{dataset_path}{name}",
                                                             sampling_rate,
                                                             sound,
                                                             extension=extension)
                if len(DataExtractor.get_circor_noisy_labels(labels)) > 0:
                    print(f"Skipping {name}.wav.\tNoisy labels.")
                    skipped += 1
                    continue
            except:
                print(f"Skipping {name}.wav\tNo label file.")
                skipped += 1
                continue
            dataset[i, 0] = f"{name}.wav"
            dataset[i, 1] = sound
            dataset[i, 2] = labels
            i += 1

        dataset = dataset[:-skipped] if skipped > 0 else dataset
        dataset[:, 1] = DataExtractor.resample_signal(dataset[:, 1], original_rate=4000, new_rate=50)
        dataset[:, 2] = DataExtractor.resample_labels(dataset[:, 2], original_rate=4000, new_rate=50)
        return dataset

    @staticmethod
    def resample_labels(labels, original_rate, new_rate):
        resampled_labels = []
        for label in labels:
            indices = np.arange(start=0, step=int(original_rate / new_rate), stop=len(label))
            resampled_labels.append(label[indices])
        return np.array(resampled_labels)

    @staticmethod
    def align_downsampled_dataset(dataset):
        for i in range(len(dataset)):
            diff = dataset[i, 2].shape[0] - dataset[i, 1].shape[0]
            if diff != 0:
                dataset[i, 2] = dataset[i, 2][:-diff]
        return dataset

    @staticmethod
    def resample_signal(data, original_rate=1000, new_rate=50):
        resampled_data = []
        for recording in data:
            time_secs = len(recording) / original_rate
            number_of_samples = int(time_secs * new_rate)
            # downsample from the filtered signal
            resampled_data.append(scipy.signal.resample(recording, number_of_samples).squeeze())
        return np.array(resampled_data)

    @staticmethod
    def get_power_spectrum(data, sampling_rate, window_length, window_overlap, window_type='hann'):
        psd_data = np.zeros(data.shape, dtype=object)
        for i in range(len(data)):
            recording = data[i]
            # Apply high-pass and low pass order 2 Butterworth filters with respective 25 and 400 Hz cut-offs
            sos_hp = scipy.signal.butter(N=2, Wn=25, btype='highpass', analog=False, fs=sampling_rate, output='sos')
            sos_lp = scipy.signal.butter(N=2, Wn=400, btype='lowpass', analog=False, fs=sampling_rate, output='sos')
            filtered = scipy.signal.sosfilt(sos_hp, recording)
            filtered = scipy.signal.sosfilt(sos_lp, filtered)
            _, _, psd = scipy.signal.stft(filtered.squeeze(),
                                          fs=sampling_rate,
                                          window=window_type,
                                          nperseg=window_length,
                                          noverlap=window_overlap)
            # transform the signal from complex to real-valued
            # Transpose to get the number of windows in first dimension to have the frequencies has a fixed
            # dimension for the CNNs
            psd = np.abs(psd).T

            # PSD_norm(t,f) = PSD(t,f)/ A; A=sum_t=0^T-1 sum_f=0^F-1 PSD(t,f)/T => sum PSD_Norm(t,f) = T
            length_psd = psd.shape[0]
            normalization = np.sum(np.sum(psd, axis=0))
            psd_data[i] = psd / (normalization / length_psd)

        return psd_data

    @staticmethod
    def get_mfccs(data, sampling_rate, window_length, window_overlap, n_mfcc, fmin=25, fmax=400, resample=None,
                  delta=True, delta_delta=True):
        if resample is not None:
            data = DataExtractor.resample_signal(data, original_rate=sampling_rate, new_rate=resample)
            sampling_rate = resample  # TODO: this is bad practice, change after test

        mfcc_data = np.zeros(data.shape, dtype=object)
        _hop_length = window_length - window_overlap
        for i in range(len(data)):
            recording = data[i]
            S = librosa.feature.melspectrogram(y=recording.squeeze(),
                                               n_fft=window_length,
                                               sr=sampling_rate,
                                               hop_length=_hop_length,
                                               fmin=fmin,
                                               fmax=fmax,
                                               window='hann')
            # Convert to log scale (dB).
            log_S = librosa.amplitude_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)
            # Transpose to [T, Mel] and normalize using Global Cepstral Mean
            mfcc = speechpy.processing.cmvn(mfcc.T, variance_normalization=True)
            if delta_delta is True:
                delta_ = librosa.feature.delta(mfcc, mode='mirror')
                delta_delta_ = librosa.feature.delta(mfcc, order=2, mode='mirror')
                mfcc_data[i] = np.concatenate([mfcc, delta_, delta_delta_], axis=1)
            elif delta is True:
                delta_ = librosa.feature.delta(mfcc, mode='mirror')
                mfcc_data[i] = np.concatenate([mfcc, delta_], axis=1)
            else:
                mfcc_data[i] = mfcc

        return mfcc_data

    @staticmethod
    def calculate_delta(coefficients, delta_diff=2):
        """
        Given coeffients of a delta^k mfcc, calculates delta^(k+1).
        Normalization according to:
        http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#deltas-and-delta-deltas
        Parameters
        ----------
        coefficients : np.ndarray
            A ndarray of shape (t, c), i.e. c coefficients for every sample t in the signal
        delta_diff : int
            The offset in time used to calculate the deltas

        Returns
        -------
        np.ndarray
            A ndarray of shape (t, c), where each t has c delta coefficients
        """
        delta = np.zeros(coefficients.shape)
        norm = 2 * np.sum(np.arange(1, delta_diff + 1) ** 2)
        for t in range(coefficients.shape[0]):
            d_t = 0
            for n in range(delta_diff):
                d_t += (n + 1) * (
                        coefficients[min(coefficients.shape[0] - 1, t + n), :] - coefficients[max(0, t - n), :]
                )
            delta[t, :] = d_t / norm
        return delta

    @staticmethod
    def extract(path, patch_size, filter_noisy=True):
        data = sio.loadmat(path, squeeze_me=True)
        raw_features = data['Feat_cell']
        raw_labels = data['Lab_cell']
        raw_patient_ids = data['Number_cell']

        # remove sounds shorter than patch size (and record sound indexes)
        length_sounds = np.array([len(raw_features[j]) for j in range(len(raw_features))])
        valid_indices = np.array([j for j in range(len(raw_features)) if len(raw_features[j]) >= patch_size])
        # Filter noisy labels. (Use for filtering out small label mistakes in Springer16)
        if filter_noisy:
            labels_ = raw_labels[valid_indices]
            noisy_indices = []
            for idx, lab in enumerate(labels_):
                lab = lab - 1
                for t in range(1, len(lab)):
                    if lab[t] != lab[t - 1] and lab[t] != (lab[t - 1] + 1) % 4:
                        noisy_indices.append(valid_indices[idx])

            valid_indices = np.array(list(set(valid_indices) - set(noisy_indices)))
            print(f"Filtered {len(set(noisy_indices))} observations containing noisy labels")
        features = raw_features[valid_indices]
        labels = raw_labels[valid_indices]
        patient_ids = raw_patient_ids[valid_indices]

        return valid_indices, features, labels, patient_ids, length_sounds

    @staticmethod
    def filter_by_index(processed_features, indices):
        return processed_features[indices]
