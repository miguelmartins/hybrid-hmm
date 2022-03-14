import librosa
import numpy as np
import scipy.io as sio
import scipy.signal
from sklearn.preprocessing import scale


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
    def downsample_signal(data, original_rate=1000, new_rate=50):
        downsampled_data = []
        for recording in data:
            time_secs = len(recording) / original_rate
            number_of_samples = int(time_secs * new_rate)
            # downsample from the filtered signal
            downsampled_data.append(scipy.signal.resample(recording, number_of_samples).squeeze())
        return np.array(downsampled_data)

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
    def get_mfccs(data, sampling_rate, window_length, window_overlap, n_mfcc, fmin=25, fmax=400):
        mfcc_data = np.zeros(data.shape, dtype=object)
        _hop_length = window_length - window_overlap
        for i in range(len(data)):
            recording = data[i]
            mfcc = librosa.feature.mfcc(recording.squeeze(),
                                        n_fft=window_length,
                                        sr=sampling_rate,
                                        hop_length=_hop_length,
                                        n_mfcc=n_mfcc,
                                        fmin=fmin,
                                        fmax=fmax)
            mfcc_data[i] = mfcc.T  # switch the time domain to the first dimension

        return mfcc_data

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
