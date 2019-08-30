
import os
import numpy as np
import pandas as pd
import string
import random

import matplotlib.pyplot as plt
import cloudpickle
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

#load data
datapickle_file = "violence_data.pkl"
print("Loading data to file %s" % datapickle_file)
(x_train, y_train, x_test, y_test) = cloudpickle.load(open(datapickle_file, 'rb'))
y_train = y_train.ravel()
y_test = y_test.ravel()
print(x_train.shape)
model = XGBClassifier(
                learning_rate=0.2,
                n_estimators=250,
                # max_depth=5,


                # subsample=0.6,
                # seed=27,
                objective="multi:softprob",
                verbosity=2
                # scale_pos_weight=0.5,
                # min_child_weight=3,
                # gamma=0.2
    )
# model = XGBClassifier(verbosity=1)
model.fit(x_train, y_train,
        eval_metric=['auc', 'error'], verbose=1
)
scores = model.score(x_test, y_test)
print(scores)
print("score train:", model.score(x_train, y_train))

preds = model.predict(x_test)
pred_proba = model.predict_proba(x_test)[:, 1]
print("F1 Score : %f" % f1_score(y_test, preds))
print("AUC : %f" % roc_auc_score(y_test, pred_proba))
