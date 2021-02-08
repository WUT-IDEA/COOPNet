import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
from sklearn import metrics


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                    # 'text_senti_fig_rvdropout')
                                    'text_senti_fig')

list = [
    'model-epoch_01_3predictions.npy',
    'model-epoch_02_3predictions.npy',
    'model-epoch_03_3predictions.npy',
    'model-epoch_04_3predictions.npy',
    'model-epoch_05_3predictions.npy',
    'model-epoch_06_3predictions.npy',
    'model-epoch_07_3predictions.npy',
    'model-epoch_08_3predictions.npy',
    'model-epoch_09_3predictions.npy',
    'model-epoch_10_3predictions.npy',
    'model-epoch_11_3predictions.npy',
    'model-epoch_12_3predictions.npy',
    'model-epoch_13_3predictions.npy',
    'model-epoch_14_3predictions.npy',
    'model-epoch_15_3predictions.npy',
    'model-epoch_16_3predictions.npy',
    'model-epoch_17_3predictions.npy',
    'model-epoch_18_3predictions.npy',
    'model-epoch_19_3predictions.npy',
    'model-epoch_20_3predictions.npy',

    'model-epoch_01+20_3predictions.npy',
    'model-epoch_02+20_3predictions.npy',
    'model-epoch_03+20_3predictions.npy',
    'model-epoch_04+20_3predictions.npy',
    'model-epoch_05+20_3predictions.npy',
    'model-epoch_06+20_3predictions.npy',
    'model-epoch_07+20_3predictions.npy',
    'model-epoch_08+20_3predictions.npy',
    'model-epoch_09+20_3predictions.npy',
    'model-epoch_10+20_3predictions.npy',
    'model-epoch_11+20_3predictions.npy',
    'model-epoch_12+20_3predictions.npy',
    'model-epoch_13+20_3predictions.npy',
    'model-epoch_14+20_3predictions.npy',
    'model-epoch_15+20_3predictions.npy',
    'model-epoch_16+20_3predictions.npy',
    'model-epoch_17+20_3predictions.npy',
    'model-epoch_18+20_3predictions.npy',
    'model-epoch_19+20_3predictions.npy',
    'model-epoch_20+20_3predictions.npy',
]


def vote():
    for filename in list:
        path = checkpoint_root_path + "/" + filename
        print(path)
        data = np.load(path)

        label = data[:, 0]
        prob = data[:, 1:4]

        prob_avg = np.sum(prob, axis=1) / 3

        pred = [1 if i>=0.5 else 0 for i in prob_avg]

        print('acc:', metrics.accuracy_score(y_true=label, y_pred=pred))
        print('auc:', metrics.roc_auc_score(y_true=label, y_score=prob_avg))

        print(metrics.confusion_matrix(y_true=label, y_pred=pred))
        print(metrics.classification_report(y_true=label, y_pred=pred))


def stack():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'ckp', 'text_senti_fig_rvdropout',
                             'model-epoch_16-pred_on_trainvalid.npy')

    train_arr = np.load(file_path)
    X_train = train_arr[:, 1:4]

    Y_train = []
    for item in train_arr[:, 0]:
        Y_train.append(item[0])
    Y_train = np.array(Y_train)
    # print(X_train)
    # print(Y_train)



    for filename in list:
        path = checkpoint_root_path + "/" + filename
        print(path)
        data = np.load(path)

        X_test = data[:, 1:4]
        Y_test = data[:, 0]
        # print(X_test)
        # print(X_test.shape)


        ########## LR
        model = LogisticRegression(penalty='l2', C=1)
        model.fit(X_train, Y_train)
        print('Accuarcy of LR ---------------------:', model.score(X_test, Y_test))

        prob = model.predict_proba(X_test)
        pred = [1 if i>=0.5 else 0 for i in prob[:, 1]]

        print('acc:', metrics.accuracy_score(y_true=Y_test, y_pred=pred))
        print('auc:', metrics.roc_auc_score(y_true=Y_test,  y_score=prob[:, 1]))

        print(metrics.confusion_matrix(y_true=Y_test, y_pred=pred))
        print(metrics.classification_report(y_true=Y_test, y_pred=pred, digits=6))



        # ########## RF
        # model = RandomForestClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5,
        #                                max_features=3, max_leaf_nodes=20
        #                                )
        # model.fit(X_train, Y_train)
        # print('Accuarcy of RF ---------------------:', model.score(X_test, Y_test))
        #
        # prob = model.predict_proba(X_test)
        # auc = metrics.roc_auc_score(y_true=Y_test,  y_score=prob[:, 0])
        # print('AUC of RF ---------------------:',auc)
        #
        # ######### RF
        # model = SVC(kernel='linear')
        # model.fit(X_train, Y_train)
        # print('Accuarcy of SVM ---------------------:', model.score(X_test, Y_test))
        #
        # prob = model.predict(X_test)
        # auc = metrics.roc_auc_score(y_true=Y_test,  y_score=prob)
        # print('AUC of SVM ---------------------:',auc)

# vote()
stack()