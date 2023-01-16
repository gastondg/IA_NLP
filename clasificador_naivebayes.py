from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from preprocessor import Processor

dataset = pd.read_csv(r'./datasets/posta/clasificado.csv', encoding='utf-8')

"""
Limpiamos los datos
"""
print("Limpiando dataset...")
limpieza = Processor()
dataset_limpio = dataset.copy()
dataset_limpio['text'] = dataset_limpio['text'].apply(limpieza.process)
dataset_limpio.head()
print("El dataset está listo")

"""
Separamos datos en entrenamiento y pruebas
"""
x = dataset_limpio['text']
y = dataset_limpio['label']
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size= 0.2, random_state=4)

print('Numero total de filas en el dataset: {}'.format(dataset_limpio.shape[0]))
print('Numero de filas en el set de entrenamiento: {}'.format(X_train.shape[0]))
print('Numero de filas en el set de testing: {}'.format(X_test.shape[0]))

print("Entrenando y buscando el mejor score de F1...")
# Lista de valores de alpha a probar
alphas = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.18,0.19,0.2,0.21, 0.3, 0.4, 0.5, 1]

# Lista de valores de n-gramas a probar
ngrams = [(1,1), (1,2), (1,3), (1,4), (1,5)]

# Inicializamos las variables para almacenar el mejor puntaje de F1 y los mejores valores de alpha y n-gramas
best_f1 = 0
best_alpha = 0
best_ngram = (1,1)

for ngram in ngrams:
    vectorizer = TfidfVectorizer(ngram_range=ngram)
    training_data = vectorizer.fit_transform(X_train)
    testing_data = vectorizer.transform(X_test)
    for alpha in alphas:
        naive_bayes = MultinomialNB(alpha=alpha)
        cv = KFold(n_splits=10, random_state=1, shuffle=True)
        scores = cross_val_score(naive_bayes, training_data, y_train, cv=cv, scoring='f1_macro')
        f1 = np.mean(scores)
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_ngram = ngram

# Usamos los mejores valores de alpha y n-gramas para entrenar el modelo final
vectorizer = TfidfVectorizer(ngram_range=best_ngram)
training_data = vectorizer.fit_transform(X_train)
testing_data = vectorizer.transform(X_test)
naive_bayes = MultinomialNB(alpha=best_alpha)
naive_bayes.fit(training_data, y_train)

# Realizamos las predicciones con el modelo final
predictions = naive_bayes.predict(testing_data)

# Imprimimos el mejor puntaje de F1 y los mejores valores de alpha y n-gramas
print("Best F1 score: ", best_f1)
print("Best alpha: ", best_alpha)
print("Best n-grams: ", best_ngram)

print("Probamos el modelo:")
ejemplos = ['es una verga', 'sos un groso hermano', 'muy rico todo', 'son unos chorros de mierda',
'prefiero una rica torta', 'es todo culpa de ellos', 'me gusta jugar al tenis','qué quilombazo']
for tuit in ejemplos:
    clasificacion = naive_bayes.predict(vectorizer.transform([tuit]))
    print(tuit, "===> ", clasificacion[0])

"""
Lo guardamos en un pickle para utilizarlo sin volver a entrenarlo
"""

import pickle

clasificador = 'clasificador_multinomial_v2.pkl'
vectorizador = 'vectorizador_tfidf_v2.pkl'

pickle.dump(naive_bayes, open( clasificador, "wb" ))
pickle.dump(vectorizer, open(vectorizador, "wb" ))
