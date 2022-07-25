from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score


# TODO: check with notebook on other repo to see if this is correct
def get_center_values(state_seq, sampling_rate=50):
    centers = []
    prev_start = 0  # start of the initial state
    for i in range(1, len(state_seq)):
        if state_seq[i] != state_seq[i - 1]:
            center = prev_start + i  # aggregation for average between the end of last and start of new
            centers.append([(center / sampling_rate) / 2, state_seq[i - 1]])  # save mean and previous state
            prev_start = i  # the new becomes the previous
    return centers


def schmidt_metrics(ground_truth, prediction, limit_tp=0.06):
    center_value = np.array(get_center_values(ground_truth))
    est_center_value = np.array(get_center_values(prediction))

    tp, fp = 0, 0
    for k in range(len(est_center_value)):
        out_center = est_center_value[k, 0]
        out_state = est_center_value[k, 1]
        distance = np.abs(out_center - center_value[:, 0])
        candidates = np.where(distance <= limit_tp)
        sum_condition_1 = np.sum(center_value[candidates, 1] == 1)
        sum_condition_2 = np.sum(center_value[candidates, 1] == 3)
        if out_state == 1 and sum_condition_1 > 0:
            tp += 1
        if out_state == 1 and sum_condition_1 == 0:
            fp += 1
        if out_state == 3 and sum_condition_2 > 0:
            tp += 1
        if out_state == 3 and sum_condition_2 == 0:
            fp += 1

    try:
        ppv = tp / (tp + fp)
    except:
        ppv = 0.0
    try:
        sensitivity = tp / (len(np.where(center_value[:, 1] == 1)[0]) + len(np.where(center_value[:, 1] == 3)[0]))
    except:
        sensitivity = 0.0
    return ppv, sensitivity


def get_metrics(gt, prediction):
    ppv, sensitivity, accuracy = [], [], []

    for sound in range(len(gt)):
        ppv_, sensitivity_ = schmidt_metrics(gt[sound], prediction[sound])
        accuracy.append(accuracy_score(gt[sound], prediction[sound]))
        ppv.append(ppv_)
        sensitivity.append(sensitivity_)
    mean_ppv = np.mean(ppv)
    mean_sensitivity = np.mean(sensitivity)
    mean_accuracy = np.mean(accuracy)
    return mean_ppv, mean_sensitivity, mean_accuracy


def get_segments(y: np.ndarray) -> np.ndarray:
    """
    Given a 1D state sequence, determines the start and end segment number for each
    contiguous state segment
    Parameters
    ----------
    y: a 1d np.ndarray containing a state sequence

    Returns
    -------
    A N_soundsX 2 X 3 matrix of the form A_i = [start_i, end_i, state_i]
    """
    segments = []
    signal_length = y.shape[0]
    for i in range(1, signal_length):
        if y[i] != y[i - 1]:
            segments.append([start, i - 1, y[i - 1]])
            start = i

    segments.append([start, signal_length - 1, y[-1]])
    return np.array(segments)


def get_centers(segments: np.ndarray) -> np.ndarray:
    """
    Given the segments start deliniations matrix, computes the center segment for each row.
    Output segment center indices are assumed to be continuous for computational purposes.
    Parameters
    ----------
    segments
        A 2D np.ndarray of the form A_i=[start_i, end_i, state_i]
    Returns
    -------
        A 2D  np.ndarray of the form B_j = [middle_point_j, state_j]
    """
    centers_ = (((segments[:, 1] - segments[:, 0]) / 2) + segments[:, 0])
    return np.stack([centers_, segments[:, 2]]).T  # transpose to get a n_segments X 2 matrix


def get_schmidt_tp_fp(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      sample_rate: int = 50,
                      threshold: float = 0.06) -> Tuple[int, int]:
    """
    Computes TP and FP for PCG pairs (y_true, y_pred), where S1 (state == 0) and S2 (state == 2)
    are assumed to be the positive predictions. See [Schmidt08] for details. Summary:
    A true positive is considered if the distance to the mid point of each segment in y_true and y_pred
    is less than a certain threshold (default 60 ms). All other instances are considered FP.
    Parameters
    ----------
    y_true The ground truth sequence
    y_pred The prediction sequence
    sample_rate The signal sample rate
    threshold The TP window threshold (defaults to 60 ms)

    Returns
    -------
        (int, int, int) A tuple containing the #tp and #fp and total (# of s1 and s2 in ground truth) respectively.
    """

    # Get center segments
    true_segment_s = get_centers(get_segments(y_true))
    pred_segment_s = get_centers(get_segments(y_pred))
    # Convert to time domain (seconds)
    true_segment_s[:, 0] = true_segment_s[:, 0] / sample_rate
    pred_segment_s[:, 0] = pred_segment_s[:, 0] / sample_rate

    # Find s1 and s2 segments
    true_segment_s1 = true_segment_s[true_segment_s[:, 1] == 0]
    true_segment_s2 = true_segment_s[true_segment_s[:, 1] == 2]
    pred_segment_s1 = pred_segment_s[pred_segment_s[:, 1] == 0]
    pred_segment_s2 = pred_segment_s[pred_segment_s[:, 1] == 2]

    def get_tp(x, y, threshold):
        # Check if match is within threshold. If so, mark index with true (1).
        mask_tp = np.where(np.abs(x[:, 0] - y[:, 0]) <= threshold, True, False)
        return mask_tp

    # Get indices of TP (center(y_pred_t) == center(y_true_t)
    mask_tp_s1 = get_tp(true_segment_s1, pred_segment_s1, threshold)  # For S1s
    mask_tp_s2 = get_tp(true_segment_s2, pred_segment_s2, threshold)  # For S2s
    tp = np.sum(mask_tp_s1) + np.sum(mask_tp_s2)  # Sum TP for all cases
    fp = len(pred_segment_s) - tp  # The remainder of the sounds are considered FP by default.
    total = len(true_segment_s1) + len(true_segment_s2)
    return tp, fp, total
