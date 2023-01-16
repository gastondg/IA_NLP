import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from preprocessor import Processor

# Cargar el archivo csv como un dataframe de pandas
data = pd.read_csv("./datasets/posta/clasificado.csv")

# Convertir las etiquetas a valores numéricos (0 = NEGATIVE, 1 = NEUTRAL, 2 = POSITIVE)
data['label'] = data['label'].map({'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2})

# Dividir los datos en conjuntos de entrenamiento y prueba
pre = Processor()
train_x, test_x, train_y, test_y = train_test_split(data['text'], data['label'], test_size=0.2)
train_x = [pre.process(text) for text in train_x]
test_x = [pre.process(text) for text in test_x]

# Tokenizar los textos
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_x)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

# Convertir el texto a secuencias de números
train_sequences = tokenizer.texts_to_sequences(train_x)
test_sequences = tokenizer.texts_to_sequences(test_x)

# Asegurar que todas las secuencias tengan la misma longitud
max_length = max([len(s.split()) for s in data['text']])
train_data = pad_sequences(train_sequences, maxlen=max_length)
test_data = pad_sequences(test_sequences, maxlen=max_length)

# Construir el modelo LSTM
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(LSTM(units=64, dropout=0.4, recurrent_dropout=0.0))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Entrenar el modelo
batch_size = 32
epochs = 10
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_data, train_y, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_y), callbacks=[early_stopping])

# Evaluar el modelo
scores = model.evaluate(test_data, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Guardar el modelo
model.save('lstm_sentiment_model.h5')

import pickle
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("max_length.pickle", "wb") as handle:
    pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("preprocessor.pickle", "wb") as handle:
    pickle.dump(pre, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# Crea una función para construir el modelo
def create_model(dropout_rate=0.0, recurrent_dropout_rate=0.0):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(LSTM(units=64, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Crea una instancia del modelo usando KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# Definir los parámetros para buscar en la grilla
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4]
recurrent_dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4]
param_grid = dict(dropout_rate=dropout_rate, recurrent_dropout_rate=recurrent_dropout_rate)

# Buscar los mejores parámetros
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train_data, train_y)

# Imprimir resultados
print("Los mejores parámetros son: {} con un puntaje de {}".format(grid_result.best_params_, grid_result.best_score_))


# Plotear la curva de aprendizaje
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotear la curva de pérdida
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
"""