from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

def train_logistic_model(X, y):
    train_data,test_data,train_label,test_label=train_test_split(X,y,random_state=1,train_size=0.7,test_size=0.3)
    model = LogisticRegression(max_iter=2000, multi_class='ovr', solver='newton-cg')
    model.fit(train_data, train_label.ravel())
    train_score = model.score(train_data,train_label)
    print("Training set score of logistic:",train_score)
    test_score = model.score(test_data,test_label)
    print("Testing set score of logistic:",test_score)
    return model

def train_logistic_model_with_cross_validation(X, y):
    model = LogisticRegression(max_iter=2000, multi_class='ovr', solver='newton-cg')

    # 10-fold cross-validation
    scores = cross_val_score(model, X, y, cv=10)

    print("Cross-Validation scores:", scores)
    print("Mean score:", np.mean(scores))
    return model