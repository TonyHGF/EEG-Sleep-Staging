import scoring

def test_test(X_test, y_test, pca, model):
    X_test = pca.transform(X_test)
    y_pred = model.predict(X_test)
    scoring.score(y_pred, y_test, 2)