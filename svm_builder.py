from sklearn.model_selection import train_test_split
from sklearn import svm

def train_svm_model(X, y):
    train_data,test_data,train_label,test_label=train_test_split(X,y,random_state=1,train_size=0.7,test_size=0.3)
    model = svm.LinearSVC(dual='auto', max_iter= 10000, tol=1e-3)
    model.fit(train_data, train_label.ravel())
    train_score = model.score(train_data,train_label)
    print("training set score:",train_score)
    test_score = model.score(test_data,test_label)
    print("testing set score:",test_score)
    return model