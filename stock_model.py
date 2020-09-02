#oparte na:
#towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

#importowanie bibliotek
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

#obiekt skalujacy dane do zakresu [0,1] - dobra praktyka przy podawaniu danych do sieci neuronowych
scaler = MinMaxScaler(feature_range=(0, 1))

#w ponizszym przykladzie zaimplementowany zostal tzw model rekurencyjny
#klasyczna siec neuronowa pozwala przewidywac 1 dzien w przod na podstawie obserwacji z N wczesniejszych dni
#(tutaj N=60)
#predykcja na wiele dni jest mozliwa dzieki podejsciu rekurencyjnemu
#czyli: liczymy przewidywana cene na dzien T, i dolaczamy ja do zbioru uczacego, co pozwala nam na przewidywanie ceny w dniu T+1

#uczenie modelu przewidujacego 1 dzien w przod
def train_1day_model(train_data, lag, lstm_width, dense_width, batch_size_, epochs_, drop_rate=0.1):
    dataset = np.array(train_data)

    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data

    x_train = []
    y_train = []
    for i in range(lag, len(train_data)-lag):
        x_train.append(train_data[(i - lag):i])
        y_train.append(train_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=lstm_width, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(drop_rate))
    model.add(LSTM(units=lstm_width, return_sequences=False))
    model.add(Dropout(drop_rate))
    model.add(Dense(units=dense_width))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=batch_size_, epochs=epochs_)

    return {"model": model, "lag": lag}

#przewidywanie cen na podstawie wyestymowanego juz modelu (tworzonego w funkcji train_1day_model)
#na kilka dni w przod
def predict_recursive(model_struct, train_data, pred_steps):
    lag = model_struct["lag"]
    model = model_struct["model"]
    train = scaler.fit_transform(train_data[-lag:])
    for _ in range(pred_steps):
        x = train[-lag:]
        x = np.array(x)
        x = np.reshape(x, (1, lag, 1))
        pred = model.predict(x)
        train = np.append(train, pred)
    result = np.reshape(train[-pred_steps:], (-1,1))
    return scaler.inverse_transform(result)

#tworzenie zbioru walidacyjnego - kazda probka walidacyjna sklada sie z N obserwacji uczacych i (pred_steps) obs. testowych
def make_val_set(train_data, val_data, lag, pred_steps):
    result = []
    train = np.array(train_data[-lag:])
    for i in range(len(val_data) - pred_steps):
        data_in = train[-lag:]
        data_in = np.reshape(data_in, (-1,1))

        data_out = val_data[i:(i+pred_steps)]
        data_out = np.reshape(data_out, (-1,1))

        train = np.append(train, val_data[i])
        result.append({"in": data_in, "out": data_out})
    return result

#walidowanie - tworzenie zbioru trajektorii faktycznych i zbioru predykcji
def validate(model, val_set):
    result = {"actuals": [], "forecasts": []}
    for val in val_set:
        pred = predict_recursive(model, val["in"], len(val["out"]))
        result["actuals"].append(val["out"])
        result["forecasts"].append(pred)
    return result