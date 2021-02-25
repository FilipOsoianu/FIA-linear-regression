import numpy as np
from sklearn.linear_model import LinearRegression
from utils import get_data, parce_data, liniar_regresion


if __name__ == "__main__":
    data = get_data('apartmentComplexData.csv')
    model = LinearRegression()
    X_train, Y_train, X_test, Y_test = parce_data(data)
    liniar_regresion(model, X_train, Y_train, X_test, Y_test)
