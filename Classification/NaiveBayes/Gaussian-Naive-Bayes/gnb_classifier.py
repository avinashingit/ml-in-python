# For building the Gaussioan Naive Bayes classifier, Iris dataset from Scikit Learn is being used.
from sklearn.datasets import load_iris


# Importing necessary scikit learn libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Function to calculate accuracy
def calculate_accuracy(actual_values, predicted_values):
    return accuracy_score(y_pred = predicted_values, y_true = actual_values)*100

# Loading Data
def load_data():
    irisData = load_iris()

    # Separating Features and Response from the data
    dataFeatures = irisData.data
    dataResponse = irisData.target
    return [dataFeatures, dataResponse]

# Splitting the data into train and test
def split_data(features, response, percentage):
    return train_test_split(features, response, test_size = percentage, random_state = 1)

def train_and_test_gnb():
    dataFeatures, dataResponse = load_data()
    train_features, test_features, train_response, test_response = split_data(dataFeatures, dataResponse, 0.4)

    # Initialize Gaussian Naive Bayes model
    nb_model = GaussianNB()

    # Train the model using the training dataset
    nb_model.fit(train_features, train_response)

    # Predict on training dataset
    training_predictions = nb_model.predict(train_features)
    train_accuracy = calculate_accuracy(predicted_values = training_predictions, actual_values = train_response)

    # Predict on testing dataset
    testing_predictions = nb_model.predict(test_features)
    test_accuracy = calculate_accuracy(predicted_values = testing_predictions, actual_values = test_response)

    print('Training accuracy is %3.2f, Testing accuracy is %3.2f' % (train_accuracy, test_accuracy))

if __name__ == "__main__":
    train_and_test_gnb()
