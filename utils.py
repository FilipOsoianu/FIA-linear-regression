import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_data(filePath):
    headers = [
        "IDK1",
        "IDK2",
        "ComplexAge",
        "TotalRooms",
        "TotalBedrooms",
        "ComplexInhabitants",
        "ApartamentsNumber",
        "IDK3",
        "MedianComplexValue"
    ]
    return pd.read_csv(filePath, names=headers)


def parce_data(data):
    Y = data.iloc[:, 8].values
    X = data.drop(['MedianComplexValue'], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    return X_train, Y_train, X_test, Y_test


def liniar_regresion(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    accuracy = model.score(X_train, Y_train)
    prediction = model.predict(X_test)

    results = pd.DataFrame({'Predicted Value': prediction, 'Actual Value': Y_test})
    results = results.reset_index()
    results = results.drop(['index'], axis=1)

    plt.plot(results[:100])
    plt.legend(['Actual Value', 'Predicted Value'])
    plt.savefig('plot.svg')

    print("Accuracy = " + "{:10.4f}".format(accuracy * 100) + ' %')
