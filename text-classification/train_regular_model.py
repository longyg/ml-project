import nltk
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import load_data
import vectorize_data

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def train_svc_model(data):
    (train_texts, train_labels), (val_texts, val_labels) = data
    x_train, x_val = vectorize_data.tfidf_vectorize(train_texts, train_labels, val_texts)

    estimator = SVC(C=300,
                    kernel='rbf',
                    gamma=1.5, 
                    shrinking=True, 
                    tol=0.001, 
                    cache_size=1000,
                    class_weight=None,
                    max_iter=-1, 
                    decision_function_shape='ovr',
                    random_state=42)
    classifier = OneVsRestClassifier(estimator, n_jobs=-1)
    classifier.fit(x_train, train_labels)
    return classifier.score(x_val, val_labels)

def train_multiple_models(data):
    (train_texts, train_labels), (val_texts, val_labels) = data
    x_train, x_val = vectorize_data.tfidf_vectorize(train_texts, train_labels, val_texts)
    # x_train = x_train.toarray()
    # x_val = x_val.toarray()

    names = [
            #  "RBF SVM", 
            #  "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"
            ]
    classifiers = [
                    # SVC(C=300,
                    #     kernel='rbf',
                    #     gamma=1.5, 
                    #     shrinking=True, 
                    #     tol=0.001, 
                    #     cache_size=1000,
                    #     class_weight=None,
                    #     max_iter=-1, 
                    #     decision_function_shape='ovr',
                    #     random_state=42),
                    # GaussianProcessClassifier(1.0 * RBF(1.0)),
                    DecisionTreeClassifier(max_depth=5),
                    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                    MLPClassifier(alpha=1),
                    AdaBoostClassifier(),
                    GaussianNB(),
                    QuadraticDiscriminantAnalysis()
                ]
    for name, clf in zip(names, classifiers):
        clf.fit(x_train, train_labels)
        score = clf.score(x_val, val_labels)
        print(name, " ===> ",  score)

if __name__ == '__main__':
    nltk.download('wordnet')
    class_names, data = load_data.load_cook_train_data(isLemmatize=True)
    train_multiple_models(data)