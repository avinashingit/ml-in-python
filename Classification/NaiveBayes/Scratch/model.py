from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import datasets
import scipy.stats


def get_data():
    iris = datasets.load_iris()
    features, target = iris.data, iris.target

    train_features, train_targets = features[:120, :], target[:120]
    test_features, test_targets = features[120:, :], target[120:]

    return train_features, train_targets, test_features, test_targets


def calculate_dist_parameters(n_labels, n_features, train_features,
                              train_targets, feature_cats):
    train_mean, train_std = \
        np.zeros((n_labels, n_features)), np.zeros((n_labels, n_features))
    for i in range(n_features):
        if feature_cats[i] == 1:
            for j in range(n_labels):
                j_features = train_features[np.where(train_targets == j), i]
                mean, std_dev = np.mean(j_features), np.std(j_features)
                train_mean[j, i] = mean
                train_std[j, i] = std_dev
    return train_mean, train_std


def calculate_likelihood(n_labels, train_features, train_targets):
    class_probabilities = np.zeros(n_labels)
    for i in range(n_labels):
        prob = len(np.where(train_targets == i)[0])
        class_probabilities[i] = prob/np.float(train_features.shape[0])
    return class_probabilities


def get_test_predictions(n_labels, test_features, train_mean, train_std,
                         class_probabilities, feature_cats, train_features,
                         train_targets):
    test_predictions = []
    for record in test_features:
        class_probs = np.ones(n_labels)
        for i in range(n_labels):
            for j in range(n_features):
                if feature_cats[i] == 1:
                    class_probs[i] *= \
                        (scipy.stats.norm(train_mean[i, j],
                                          train_std[i, j]).pdf(record[j]))
                else:
                    req_features = train_features[np.where(train_targets == i),
                                                  j]
                    prob = len(np.where(req_features == record[j])[0]) / \
                        len(req_features)
                    class_probs[i] *= prob
            class_probs[i] *= class_probabilities[i]
        test_predictions.append(np.argmax(class_probs))
    return test_predictions


def calculate_accuracy(truth, predicted):
    return np.sum(truth == predicted)/float(len(truth))


if __name__ == "__main__":

    train_features, train_targets, test_features, test_targets = get_data()

    n_features = train_features.shape[1]
    n_labels = len(set(train_targets))

    # 1 - continuous, 0 - categorical
    feature_cats = [1 if len(set(train_features[:, i])) >
                    10 else 0 for i in range(train_features.shape[1])]

    train_mean, train_std = calculate_dist_parameters(n_labels, n_features,
                                                      train_features,
                                                      train_targets,
                                                      feature_cats)
    class_probabilities = calculate_likelihood(n_labels, train_features,
                                               train_targets)

    test_predictions = get_test_predictions(n_labels, test_features, train_mean,
                                            train_std, class_probabilities,
                                            feature_cats, train_features,
                                            train_targets)

    print("The accuracy through my code is {}".format(
        calculate_accuracy(test_predictions, test_targets)))

    # Cross check
    nb_model = GaussianNB()

    # Train the model using the training dataset
    nb_model.fit(train_features, train_targets)

    # print(nb_model.class_prior_)
    # print(nb_model.theta_)
    # print(np.sqrt(nb_model.sigma_))

    testing_predictions = nb_model.predict(test_features)
    print("The accuracy through GNB model is {}".format(
        calculate_accuracy(testing_predictions, test_targets)))
