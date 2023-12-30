from sklearn.model_selection import train_test_split
from sklearn import svm

def train_svm_model(X, y):
    train_data,test_data,train_label,test_label=train_test_split(X,y,random_state=1,train_size=0.7,test_size=0.3)
    model = svm.SVC(kernel="linear", decision_function_shape="ovr")
    model.fit(train_data, train_label.ravel())
    train_score = model.score(train_data,train_label)
    print("训练集：",train_score)
    test_score = model.score(test_data,test_label)
    print("测试集：",test_score)