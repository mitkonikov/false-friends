import collections

from sklearn import pipeline, preprocessing
from sklearn.impute import SimpleImputer

def build_classifier(base_clf):
    # The imputer is for "use_taxonomy", and shouldn't affect if it's False.
    # TODO: should also try with other imputer strategies
    return pipeline.make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        preprocessing.StandardScaler(),
        base_clf,
    )

def calculate_measures(tn, fp, fn, tp):
    return collections.OrderedDict(
        [
            ("tn", tn),
            ("fp", fp),
            ("fn", fn),
            ("tp", tp),
            ("Neg. Precision", tn / (tn + fn) if tn + fn > 0 else 0),
            ("Neg. Recall", tn / (tn + fp) if tn + fp > 0 else 0),
            ("Neg. F1-score", tn / (tn + (fp + fn) / 2) if tn + fn + fp > 0 else 0),
            ("Precision", tp / (tp + fp) if tp + fp > 0 else 0),
            ("Recall", tp / (tp + fn) if tp + fn > 0 else 0),
            ("F1-score", tp / (tp + (fp + fn) / 2) if tp + fp + fn > 0 else 0),
            (
                "Accuracy",
                (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn > 0 else 0,
            ),
        ]
    )

def classify(X_train, X_test, y_train, y_test, clf):
    print("Classifying...")
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)

    tn = sum(
        1 for test, predicted in zip(y_test, y_predicted) if not test and not predicted
    )
    fp = sum(
        1 for test, predicted in zip(y_test, y_predicted) if not test and predicted
    )
    fn = sum(
        1 for test, predicted in zip(y_test, y_predicted) if test and not predicted
    )
    tp = sum(1 for test, predicted in zip(y_test, y_predicted) if test and predicted)

    return calculate_measures(tn, fp, fn, tp)
