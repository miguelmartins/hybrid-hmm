import tensorflow as tf
import numpy as np
import os
import datetime

from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from custom_train_functions.hmm_train_step import train_HMM_parameters, hmm_train_step, hmm_train_step_nn_only
from data_processing.signal_extraction import DataExtractor
from loss_functions.MMI_losses import MMILossUnet
from models.custom_models import unet_pcg
from data_processing.data_transformation import PCGDataPreparer, HybridPCGDataPreparer, \
    unet_prepare_validation_data, get_data_from_generator, get_train_test_indices
from utility_functions.experiment_logs import PCGExperimentLogger
from utility_functions.hmm_utilities import log_viterbi_no_marginal
from utility_functions.metrics import get_metrics
from tensorflow.keras.utils import Progbar


def main():
    patch_size = 64
    stride = 8
    nch = 4
    number_folders = 10
    learning_rate = 1e-4
    EPOCHS = 5
    BATCH_SIZE = 1

    dp = PCGDataPreparer(patch_size=patch_size, stride=stride, number_channels=nch, num_states=4)
    dp_dev = PCGDataPreparer(patch_size=patch_size, stride=stride, number_channels=nch, num_states=4)

    good_indices, features, labels, patient_ids, length_sounds = DataExtractor.extract(path='../datasets/PCG'
                                                                                            '/PhysioNet_SpringerFeatures_Annotated_featureFs_50_Hz_audio_ForPython.mat',
                                                                                       patch_size=patch_size)

    print('Total number of valid sounds with length > ' + str(patch_size / 50) + ' seconds: ' + str(len(good_indices)))

    # 10-fold cross validation

    experiment_logger = PCGExperimentLogger(path='../results/unet', name='hybrid_unet', number_folders=number_folders)

    model = unet_pcg(nch, patch_size)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.save_weights('random_init_unet')
    loss_object = MMILossUnet(tf.Variable(tf.zeros((4, 4)), trainable=True, dtype=tf.float32),
                              tf.Variable(tf.zeros((4,)), trainable=True, dtype=tf.float32))

    acc_folds, prec_folds = [], []
    for fold in range(number_folders):
        min_val_loss = 1e3
        model.load_weights('random_init_unet')
        train_indices, test_indices = get_train_test_indices(good_indices=good_indices,
                                                             number_folders=number_folders,
                                                             patient_ids=patient_ids,
                                                             fold=fold)

        print('Considering folder number:', fold + 1)
        features_train = features[train_indices]
        features_test = features[test_indices]

        labels_train = labels[train_indices]
        labels_test = labels[test_indices]

        # This ia residual code from the initial implementation, kept for "panicky" reasons
        train_indices = good_indices[train_indices]
        test_indices = good_indices[test_indices]

        print('Number of training sounds:', len(labels_train))
        print('Number of testing sounds:', len(labels_test))
        X_train, X_dev, y_train, y_dev = train_test_split(
            features_train, labels_train, test_size=0.1, random_state=42)

        print("Shape: ", X_train.shape)

        dp.set_features_and_labels(X_train, y_train)

        train_dataset = get_data_from_generator(data_processor=dp,
                                                batch_size=BATCH_SIZE,
                                                patch_size=patch_size,
                                                number_channels=nch,
                                                number_classes=4,
                                                trainable=False)  # I am not sure about the shuffling, might remove it

        dp_dev.set_features_and_labels(X_dev, y_dev)
        dev_dataset = get_data_from_generator(data_processor=dp_dev,
                                              batch_size=BATCH_SIZE,
                                              patch_size=patch_size,
                                              number_channels=nch,
                                              number_classes=4,
                                              trainable=False)

        dp_test = PCGDataPreparer(patch_size=patch_size, stride=stride, number_channels=nch, num_states=4)
        dp_test.set_features_and_labels(features_test, labels_test)

        # Use the HMM DataPreparer to extract the labels for the HMM MLE
        hmm_dp = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
        hmm_dp.set_features_and_labels(X_train, y_train)
        train_dataset_ = tf.data.Dataset.from_generator(hmm_dp,
                                                        output_signature=(
                                                            tf.TensorSpec(shape=(None, patch_size, nch),
                                                                          dtype=tf.float32),
                                                            tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                        )

        dataset_np = list(train_dataset_.as_numpy_iterator())
        dataset = np.array(dataset_np, dtype=object)
        labels_ = dataset[:, 1]
        print(y_train.shape, labels_.shape)

        p_states, trans_mat = train_HMM_parameters(labels_)
        loss_object.trans_mat.assign(tf.Variable(trans_mat, trainable=True, dtype=tf.float32))
        loss_object.p_states.assign(tf.Variable(p_states, trainable=True, dtype=tf.float32))
        # train_dataset = train_dataset.shuffle(len(X_train), reshuffle_each_iteration=True)
        for ep in range(EPOCHS):
            print('=', end='')
            pb_i = Progbar(None)
            for (x_train, y_train) in train_dataset:
                hmm_train_step_nn_only(model=model,
                                       optimizer=optimizer,
                                       loss_object=loss_object,
                                       train_batch=x_train,
                                       label_batch=y_train,
                                       metrics=None)
                pb_i.add(1)

            # check performance at each epoch
            (train_loss, train_acc) = model.evaluate(train_dataset, verbose=0)
            print("Train Loss", train_loss, "Train_acc", train_acc)
            (val_loss, val_acc) = model.evaluate(dev_dataset, verbose=0)
            print("Epoch ", ep, " train loss = ", train_loss, " train accuracy = ", train_acc, "val loss = ", val_loss,
                  "val accuracy = ", val_acc)
            checkpoint = './checkpoints/hybrid_physion/' + str(fold) + '/my_checkpoint'
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # SAVE MODEL
                best_p_states = loss_object.p_states.numpy()
                best_trans_mat = loss_object.trans_mat.numpy()
                experiment_logger.save_model_checkpoints(model, best_p_states, best_trans_mat, '/cnn_weights_fold_',
                                                         fold)

        model = experiment_logger.load_model_checkpoint_weights(model)
        loss_object.p_states.assign(tf.Variable(best_p_states, trainable=True, dtype=tf.float32))
        loss_object.trans_mat.assign(tf.Variable(best_trans_mat, trainable=True, dtype=tf.float32))

        # model_checkpoint = ModelCheckpoint(filepath=experiment_logger.path + '/weights.hdf5', monitor='val_loss', save_best_only=True)
        # TODO: change this checkpoint maybe
        model.save(experiment_logger.path + '/my_model.h5')  # creates a HDF5 file 'my_model.h5'

        # prediction on test data
        val_dataset = get_data_from_generator(data_processor=dp_test,
                                              batch_size=BATCH_SIZE,
                                              patch_size=patch_size,
                                              number_channels=nch,
                                              number_classes=4,
                                              trainable=False)

        out_test = model.predict(val_dataset)

        hmm_dp_test = HybridPCGDataPreparer(patch_size=patch_size, number_channels=nch, num_states=4)
        hmm_dp_test.set_features_and_labels(features_test, labels_test)
        test_dataset = tf.data.Dataset.from_generator(hmm_dp_test,
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, patch_size, nch),
                                                                        dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
                                                      )
        # recover sound labels from patch labels
        output_probs, output_seqs = unet_prepare_validation_data(out_test, test_indices, length_sounds, patch_size,
                                                                 stride)
        accuracy, precision = [], []
        i = 0
        labels_list, predictions_list = [], []

        for x, y in test_dataset:
            logits = output_probs[i]
            y = y.numpy()
            _, _, predictions = log_viterbi_no_marginal(p_states, trans_mat, logits)
            predictions = predictions.astype(np.int32)
            labels_ = np.argmax(y, axis=1).astype(np.int32)
            predictions_list.append(predictions)
            labels_list.append(labels_)
            acc = accuracy_score(labels_, predictions)
            prc = precision_score(labels_, predictions, average=None)
            accuracy.append(acc)
            precision.append(prc)
            i += 1
        print("Testing shape", output_seqs.shape, "number of x in test_dataset", i, "labels shape", len(labels_list))
        print("Mean Test Accuracy: ", np.mean(accuracy), "Mean Test Precision: ", np.mean(precision))
        acc_folds.append(np.mean(accuracy))
        prec_folds.append(np.mean(precision))
        sample_acc = np.zeros((len(labels_test),))
        for j in range(len(labels_test)):
            sample_acc[j] = 1 - (np.sum((output_seqs[j] != labels_test[j] - 1).astype(int)) / len(labels_test[j]))

        print('Sounds mean sample accuracy for this folder:', np.sum(sample_acc) / len(sample_acc))
        ppv, sens, acc = get_metrics(labels_list, output_seqs)

        # collecting data and results
        experiment_logger.update_results(fold=fold,
                                         train_indices=train_indices,
                                         test_indices=test_indices,
                                         output_seqs=output_seqs,
                                         predictions=np.array(predictions_list, dtype=object),
                                         ground_truth=np.array(labels_list, dtype=object))

    experiment_logger.save_results(p_states=p_states, trans_mat=trans_mat)


if __name__ == "__main__":
    main()
