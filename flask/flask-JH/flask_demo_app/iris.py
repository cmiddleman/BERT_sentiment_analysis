import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle

# For eda: X will have column names
X, y = datasets.load_iris(return_X_y=True, as_frame=True)

iris = datasets.load_iris()
X_modeling = iris.data
y_modeling = iris.target

def eda_graphing():
    '''
    Graphs all 4 columns in the iris dataset against each other
    
    PARAMETERS
    ----------
        None
    
    RETURNS
    -------
        None
    '''
    for i, col1 in enumerate(X.columns):
        for j, col2 in enumerate(X.columns):
            if col1 == col2:
                pass            
            else:
                x_min, x_max = X.iloc[:, i].min() - .5, X.iloc[:, i].max() + .5
                y_min, y_max = X.iloc[:, j].min() - .5, X.iloc[:, j].max() + .5

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X.iloc[:, i], X.iloc[:, j], c=y, cmap=plt.cm.Set1,
                        edgecolor='k')
                ax.set_xlabel(f'{col1}')
                ax.set_ylabel(f'{col2}')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks(())
                ax.set_yticks(())

                # Note: saving to static/images bc these images will not change
                fig.savefig(f'./static/images/col{i}col{j}.png')
                fig.show();

def logistic_regression(X, y):
    '''
    Trains a logistic regression to predict target column
    from iris dataset. Pickles model for later use in app.py.

    PARAMETERS
    ----------
        X: iris data (4 columns) numpy array
        y: iris target, numpy array

    RETURNS
    -------
        None
    '''

    # Train test split
    test_size = 0.2
    seed = 2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Instantiate model and train on training data
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model on holdout data
    y_pred = model.predict(X_test)

    # Print results
    print(classification_report(y_test, y_pred))
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')

    # Pickle model
    filename = 'iris_log_regr.pkl'
    pickle.dump(model, open(filename, 'wb'))

    # Load model and test that result is the same as accuracy score above
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print(f'Loaded model results: {result}')

if __name__ == "__main__":
    # Graph all iris dataset columns against each other
    # Saves to images folder in static
    eda_graphing()

    # Train logistic regression and pickle model
    # Prints model test results and loaded model test results
    logistic_regression(X_modeling, y_modeling)  