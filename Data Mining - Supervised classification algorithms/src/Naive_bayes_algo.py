import csv
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB as NB


# Function for 10 fold cross validation
   
def cross_validation(model, _X, _Y, _cv=10):
    '''Function to perform 10 Folds Cross-Validation
    Parameters
    ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    _X: array
        This is the matrix of features.
    _y: array
        This is the target variable.
    _cv: int, default=5
        Determines the number of folds for cross-validation.
    Returns
    -------
    The function returns a dictionary containing the metrics 'accuracy', 'precision',
    'recall', 'f1' for both training set and validation set.
    '''
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                            X=_X,
                            y=_Y,
                            cv=_cv,
                            scoring=_scoring,
                            return_train_score=True)
    
    return {
            "Mean Training Accuracy": results['train_accuracy'].mean()*100,            
            "Mean Training Precision": results['train_precision'].mean(),            
            "Mean Training Recall": results['train_recall'].mean(),
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
            "Mean Validation Precision": results['test_precision'].mean(),
            "Mean Validation Recall": results['test_recall'].mean(),
            "Mean Validation F1 Score": results['test_f1'].mean()
            }

#Function for writing predicted results into a csv file

def output_writer(path, result):
    with open(path, 'w') as f:
        file = csv.writer(f, delimiter=',', quotechar='\r')
        for item in result:
            file.writerow([int(item)])
    print('Results have been successfully saved to: %s' % (path))


def main(): 

    # Reading and importing the data file
    data = pd.read_csv('../data/mushroom.csv')
    

    # Printing dimentions of the input data file
    print("\nThe dataset contains " + str(data.shape[0]) + " rows and " + str(data.shape[1]) + " columns")
    

    # Pre-processing the input data
    # Separating the response from predictors
    # 'class' is reponse in this dataset and and rest columns are predictors
    X = data.drop(['class'],axis=1) 
    Y = data['class']

    #Encoding the catagorial data via Label Encoding
    X = pd.get_dummies(X)
    encoder= LabelEncoder()
    Y = encoder.fit_transform(Y)

      
    # Splitting data into training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,random_state = 42)

    #Using Naive Bayes Classifier from Sklearn library
    classifier = NB()
   
    # Fitting Decision Tree classifier to the Training set
    classifier.fit(X_train,y_train)
    
    # Predicting the values for test dataset
    y_pred=classifier.predict(X_test)

    # Writing the predicted values to a file
    print("\n")
    output_writer("../result/Naive_Bayes_Prediction_Result.csv", y_pred)
    print("\n")
        
    # Performing 10 fold cross validation for accuracy
    naive_bayes_result = cross_validation(classifier, X, Y, 10)
    for key,value in naive_bayes_result.items():
        print(key,value)

    # Creating confusion matrix for test dataset
    print("\n Confusion Matrix\n")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("\n")

main()