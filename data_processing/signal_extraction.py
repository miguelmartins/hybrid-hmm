import numpy as np
import scipy.io as sio


class DataExtractor:

    @staticmethod
    def extract(path, patch_size):
        data = sio.loadmat(path, squeeze_me=True)
        raw_features = data['Feat_cell']
        raw_labels = data['Lab_cell']
        raw_patient_ids = data['Number_cell']

        # remove sounds shorter than patch size (and record sound indexes)
        valid_indices = np.array([j for j in range(len(raw_features)) if len(raw_features[j]) >= patch_size])

        features = raw_features[valid_indices]
        labels = raw_labels[valid_indices]
        patient_ids = raw_patient_ids[valid_indices]

        return valid_indices, features, labels, patient_ids
