from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
# just for testing
def train_bayes_model(X, y):
    train_data,test_data,train_label,test_label=train_test_split(X,y,random_state=1,train_size=0.7,test_size=0.3)
    model = GaussianNB()
    model.fit(train_data, train_label.ravel())
    train_score = model.score(train_data,train_label)
    print("Training set score of Bayes:",train_score)
    test_score = model.score(test_data,test_label)
    print("Testing set score of Bayes:",test_score)
    return model