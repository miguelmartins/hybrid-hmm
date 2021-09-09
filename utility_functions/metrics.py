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
