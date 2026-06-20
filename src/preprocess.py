import pandas as pd
from sklearn.decomposition import PCA
import pickle


def load_and_preprocess():

    X_train = pd.read_csv('../data/UCI-HAR Dataset/train/X_train.txt', sep=r'\s+', header=None)
    y_train = pd.read_csv('../data/UCI-HAR Dataset/train/y_train.txt', header=None)
    X_test = pd.read_csv('../data/UCI-HAR Dataset/test/X_test.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv('../data/UCI-HAR Dataset/test/y_test.txt', header=None)

    label_counts = {}
    columns_name = []

    with open('../data/UCI-HAR Dataset/features.txt', 'r') as file:
        for line in file:
            line = line.rstrip("\n")
            parts = line.split(" ", 1)

            if len(parts) > 1:
                label = parts[1]

                if label in label_counts:
                    label_counts[label] += 1
                    unique_label = f"{label}_{label_counts[label]}"
                else:
                    label_counts[label] = 0
                    unique_label = label

                columns_name.append(unique_label)

    X_train.columns = columns_name
    X_test.columns = columns_name

    y_train[0] = y_train[0] - 1
    y_test[0] = y_test[0] - 1

    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    with open("../models/pca.pkl", "wb") as file:
        pickle.dump(pca, file)

    return X_train_pca, X_test_pca, y_train, y_test