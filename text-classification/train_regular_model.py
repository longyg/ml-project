from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import load_data
import vectorize_data

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

if __name__ == '__main__':
    class_names, data = load_data.load_cook_train_data('E:\\workspace\\notebook\\ml-project\\cook-prediction\\train.json', isLemmatize=True)
    score = train_svc_model(data)
    print(score)