import numpy as np

EPSILON = 1e-20  # 1e-12


def viterbi(p_states, trans_mat, y_pred, p_obs=None):
    T = y_pred.shape[0]
    num_states = p_states.shape[0]
    viterbi = np.zeros(y_pred.shape)
    trellis = np.zeros(y_pred.shape)
    states_aux = np.arange(num_states)
    trellis_update_fn = (lambda x, y: (x - y) % num_states)
    p_bar = p_states
    for i in range(10):
        p_bar = np.dot(p_bar, trans_mat)
    y_pred_ = np.divide(y_pred * p_obs, np.tile(p_bar, (T, 1)))
    viterbi[0, :] = p_bar * y_pred[0, :]
    for t in range(1, T):
        viterbi_lhs = np.multiply(viterbi[t - 1, :], np.diagonal(trans_mat)) * y_pred_[t, :]
        viterbi_rhs = np.multiply(np.roll(viterbi[t - 1, :], 1), np.diagonal(np.roll(trans_mat, 1, axis=0))) * y_pred_[
                                                                                                               t, :]
        viterbi[t, :] = np.maximum(viterbi_lhs, viterbi_rhs)

        l2r_prev = np.argmax(np.vstack((viterbi_lhs, viterbi_rhs)), axis=0)
        trellis[t - 1, :] = trellis_update_fn(states_aux, l2r_prev)

    best_path = np.zeros(y_pred.shape[0])
    best_path[T - 1] = int(np.argmax(viterbi[T - 1, :]))
    for t in range(T - 2, -1, -1):
        best_path[t] = int(trellis[t, int(best_path[t + 1])])
    return viterbi, trellis, best_path


def log_viterbi(p_states, trans_mat, y_pred, p_obs):
    T = y_pred.shape[0]
    num_states = p_states.shape[0]
    viterbi = np.zeros(y_pred.shape)
    trellis = np.zeros((y_pred.shape[0] - 1, y_pred.shape[1]))
    states_aux = np.arange(num_states)
    trellis_update_fn = (lambda x, y: (x - y) % num_states)
    p_bar = p_states
    for i in range(10):
        p_bar = np.dot(p_bar, trans_mat)
    p_bar = np.log(p_bar + EPSILON)
    y_pred_ = np.log(y_pred + EPSILON) + np.log(p_obs + EPSILON) - np.tile(p_bar, (T, 1))
    viterbi[0, :] = p_bar + y_pred[0, :]
    trans_mat = np.log(trans_mat + EPSILON)
    for t in range(1, T):
        viterbi_lhs = viterbi[t - 1, :] + np.diagonal(trans_mat) + y_pred_[t, :]
        viterbi_rhs = np.roll(viterbi[t - 1, :], 1) + np.diagonal(np.roll(trans_mat, 1, axis=0)) + y_pred_[t, :]
        viterbi[t, :] = np.maximum(viterbi_lhs, viterbi_rhs)

        l2r_prev = np.argmax(np.vstack((viterbi_lhs, viterbi_rhs)), axis=0)
        trellis[t - 1, :] = trellis_update_fn(states_aux, l2r_prev)

    best_path = np.zeros(y_pred.shape[0])
    best_path[T - 1] = int(np.argmax(viterbi[T - 1, :]))
    for t in range(T - 2, -1, -1):
        best_path[t] = int(trellis[t, int(best_path[t + 1])])
    return viterbi, trellis, best_path


def log_viterbi_no_marginal(p_states, trans_mat, y_pred):
    T = y_pred.shape[0]
    num_states = p_states.shape[0]
    viterbi = np.zeros(y_pred.shape)
    trellis = np.zeros((y_pred.shape[0] - 1, y_pred.shape[1]))
    states_aux = np.arange(num_states)
    trellis_update_fn = (lambda x, y: (x - y) % num_states)
    p_bar = p_states
    for i in range(10):
        p_bar = np.dot(p_bar, trans_mat)
    p_bar = np.log(p_bar + EPSILON)
    y_pred_ = np.log(y_pred + EPSILON) - np.tile(p_bar, (T, 1))
    viterbi[0, :] = p_bar + y_pred[0, :]
    trans_mat = np.log(trans_mat + EPSILON)
    for t in range(1, T):
        viterbi_lhs = viterbi[t - 1, :] + np.diagonal(trans_mat) + y_pred_[t, :]
        viterbi_rhs = np.roll(viterbi[t - 1, :], 1) + np.diagonal(np.roll(trans_mat, 1, axis=0)) + y_pred_[t, :]
        viterbi[t, :] = np.maximum(viterbi_lhs, viterbi_rhs)

        l2r_prev = np.argmax(np.vstack((viterbi_lhs, viterbi_rhs)), axis=0)
        trellis[t - 1, :] = trellis_update_fn(states_aux, l2r_prev)

    best_path = np.zeros(y_pred.shape[0])
    best_path[T - 1] = int(np.argmax(viterbi[T - 1, :]))
    for t in range(T - 2, -1, -1):
        best_path[t] = int(trellis[t, int(best_path[t + 1])])
    return viterbi, trellis, best_path


def viterbi_pobs(p_states, trans_mat, y_pred, p_obs):
    T = y_pred.shape[0]
    num_states = p_states.shape[0]
    viterbi = np.zeros(y_pred.shape)
    trellis = np.zeros(y_pred.shape)
    possible_paths = np.zeros((2, y_pred.shape[1]))
    states_aux = np.arange(num_states)
    trellis_update_fn = (lambda x, y: (x - y) % num_states)
    p_bar = p_states
    # for i in range(10):
    #     p_bar = np.dot(p_bar, trans_mat) # p' = pT
    y_pred_ = np.divide(y_pred * p_obs, np.tile(p_bar, (T, 1)))  # [p(s|o) p(o)] / p(s) [n_obs, 4]
    viterbi[0, :] = p_bar * y_pred[0, :]
    for t in range(1, T):
        possible_paths[0, :] = np.multiply(viterbi[t - 1, :], np.diagonal(trans_mat)) * y_pred_[t, :]
        possible_paths[1, :] = np.multiply(np.roll(viterbi[t - 1, :], 1),
                                           np.diagonal(np.roll(trans_mat, 1, axis=0))) * y_pred_[t, :]
        viterbi[t, :] = np.maximum(possible_paths[0], possible_paths[1])

        l2r_prev = np.argmax(possible_paths, axis=0)
        trellis[t - 1, :] = trellis_update_fn(states_aux, l2r_prev)

    best_path = np.zeros(y_pred.shape[0])
    best_path[T - 1] = int(np.argmax(viterbi[T - 1, :]))
    for t in range(T - 2, -1, -1):
        best_path[t] = int(trellis[t, int(best_path[t + 1])])
    return viterbi, trellis, best_path
