import os
import sys
import numpy as np
from sklearn import preprocessing


def get_max_normalize(features):

        maximum = max(features)
        norm_features = []

        for feature in features:
            norm_feature = feature/maximum
            norm_features.append(norm_feature)

        return norm_features


def get_l2_normalize(features):
        np_features = np.asarray(features)
        #print (np_features)
        np_features[np.isnan(np_features)] = 0.0
        X_normalized = preprocessing.normalize(np_features.reshape(1,-1), norm='l2')
        return X_normalized[0]
