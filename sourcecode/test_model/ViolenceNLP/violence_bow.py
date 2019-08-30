from load_violence_text import load_csv
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from xgboost import XGBClassifier


def sent2word(sents):
    sents = " ".join(sents)
    words = sents.split()
    return words

def view_counter():
    sents0 = load_csv("violence-data/label_0.csv")
    sents1 = load_csv("violence-data/label_1.csv")
    sents2 = load_csv("violence-data/label_2.csv")

    words1 = sent2word(sents1)
    c1 = Counter(words1)
    print(c1)

    words2 = sent2word(sents2)
    c2 = Counter(words2)
    print(c2)

def split_train_test(sents, label_id, train_ratio=0.7):

    split_id = round(len(sents)*train_ratio)
    x_train = sents[:split_id]
    x_test = sents[split_id:]
    y_train = [label_id] *len(x_train)
    y_test = [label_id] *len(x_test)
    return x_train, y_train, x_test, y_test

def run():
    x_train =[]
    y_train =[]
    x_test = []
    y_test = []

    labels = [0, 1, 2]
    for label in labels:
        fname = "violence-data/label_{}.csv".format(label)
        sents = load_csv(fname)
        print("Label: {}, len: {}".format(label, len(sents)))
        trainx, trainy, testx, testy = split_train_test(sents, label)
        x_train = x_train + trainx
        y_train = y_train + trainy
        x_test = x_test + testx
        y_test = y_test + testy

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    x_test = vectorizer.transform(x_test)
    x_test = x_test.toarray()
    print(x_train.shape)
    # pca = PCA(n_components=1000)
    # x_train = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)
    # print(x_train.shape)

    model = XGBClassifier(
                    learning_rate=0.1,
                    n_estimators=50,
                    # max_depth=5,


                    # subsample=0.6,
                    # seed=27,
                    objective="multi:softprob",
                    verbosity=2
                    # scale_pos_weight=0.5,
                    # min_child_weight=3,
                    # gamma=0.2
        )
    # model = XGBClassifier(verbosity=1)
    model.fit(x_train, y_train,
            eval_metric=['auc', 'error'], verbose=1
    )
    scores = model.score(x_test, y_test)
    print(scores)
    print("score train:", model.score(x_train, y_train))


if __name__ == '__main__':
    run()
