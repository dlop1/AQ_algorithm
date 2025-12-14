import sys
from AQ_alg import AQ
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    m = int(sys.argv[1])
    random_seed_choice = True if int(sys.argv[2]) == 1 else False
    id = int(sys.argv[3])
    data = fetch_ucirepo(id=id)
    X = data.data.features
    Y = data.data.targets

    X = X.to_numpy()
    Y = Y.to_numpy()

    domains = list([set() for _ in range(len(X[0]))])
    for x in X:
        for i, att in enumerate(x):
            domains[i].add(att)
    dom = [list(domain) for domain in domains]

    aq = AQ(m, random_seed_choice, dom)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    aq.get_data(X_train, y_train)
    aq.create_rules()

    correct_classification = 0
    for i, (x, y) in enumerate(zip(X_test, y_test)):
        result = aq.classify(x)
        if result == y:
            correct_classification += 1

    print(f'Classification quality: {correct_classification/len(X_test)*100}%')
