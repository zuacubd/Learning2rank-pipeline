import os
import sys
import pyltr
import pickle

def training_ltr_model(ltr_train_file_path, ltr_validation_file_path, ltr_model_file_path):

        with open(ltr_train_file_path) as trainfile, \
                open(ltr_validation_file_path) as valifile:
            TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
            VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)


        metric = pyltr.metrics.NDCG(k=20)
        # Only needed if you want to perform validation (early stopping & trimming)
        monitor = pyltr.models.monitors.ValidationMonitor(
                VX, Vy, Vqids, metric=metric, stop_after=250)

        model = pyltr.models.LambdaMART(
                metric=metric,
                n_estimators=500,
                learning_rate=0.02,
                max_features=0.5,
                query_subsample=0.5,
                max_leaf_nodes=10,
                min_samples_leaf=32,
                verbose=1,
        )
        model.fit(TX, Ty, Tqids, monitor=monitor)
        pickle.dump(model, open(ltr_model_file_path, 'wb'))

def loading_ltr_model(ltr_model_file_path):

    model = pickle.load(open(ltr_model_file_path, 'rb'))
    return model
