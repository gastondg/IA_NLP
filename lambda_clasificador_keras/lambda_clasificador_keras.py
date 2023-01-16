import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from preprocessor import Processor
import utils_clasificador

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("max_length.pickle", "rb") as handle:
    max_length = pickle.load(handle)

# cargar el modelo guardado
model = load_model("lstm_sentiment_model.h5")
pre = Processor()
labels = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

def classify_text(text):
    # preprocesar el texto
    text = pre.process(text)
    # convertir el texto a una secuencia de números
    sequences = tokenizer.texts_to_sequences([text])
    # asegurar que la secuencia tenga la misma longitud que las secuencias utilizadas para entrenar el modelo
    data = pad_sequences(sequences, maxlen=max_length)
    # hacer una predicción
    prediction = model.predict(data)
    # devolver la clase con la probabilidad más alta
    return labels[np.argmax(prediction)]



def handler_insert(record, stopwords, table):
    print('Analizando un nuevo registro')
    
    #Obtengo los datos del registro
    content = utils_clasificador.datos_registro(record)
    
    try:
        # limpia y prepara el texto
        clean_text = pre.process(content['text'])
        # obtengo nube y clasifico el texto
        content['nube'] = utils_clasificador.get_nube(clean_text, stopwords)
        #logger.info("Clasificando registro")
        #logger.info(content['text'], str(content['label']))
        content['label'] = classify_text(clean_text)
        
        print("Insertando: ")
        print(content["text"], "===> ", content["label"])
        
        response = table.put_item(Item=content)
        print("Status Code: ", str(response.status_code))
        print("Response:")
        print(response)

    except Exception as e:
        print('Error ',str(e))

    return response


def lambda_handler(event, context):
    #logger.info("Comenzando a procesar los registros")
    try:
        stopwords = utils_clasificador.get_stopwords()
        #clasificador = init_clasificador()
        table = utils_clasificador.get_table_dynamodb()
        for record in event['Records']:
            if record['eventName'] == 'INSERT':
                #logger.info(event)
                handler_insert(record, stopwords, table)
                #handler_insert(record)

    except Exception as e:
        print(e)
        return "Error insertando!"+ str(e)



"""
if __name__ == '__main__':
    test_cases = ["Me encanta este producto", "No estoy muy seguro de esta compra", "No puedo creer lo malo que es este producto", "Este servicio es increíble", "Estoy muy decepcionado con esta empresa"]
    labels = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    #model = load_model('lstm_sentiment_model.h5')
    #with open('tokenizer.pickle', 'rb') as handle:
    #    tokenizer = pickle.load(handle)
    for test in test_cases:
        sequence = tokenizer.texts_to_sequences([pre.process(test)])
        padded = pad_sequences(sequence, maxlen=max_length)
        prediction = model.predict(padded)
        print(f"{test} => {labels[np.argmax(prediction)]}")

    for test in test_cases:
        print("clasificación", classify_text(test))
    """

"""
import json
import boto3

def lambda_handler(event, context):
    # obtener el texto a clasificar de la solicitud
    text = json.loads(event["body"])["text"]
    # clasificar el texto
    classification = classify_text(text)
    # devolver la clasificación en la respuesta
    return {
        "statusCode": 200,
        "body": json.dumps({"classification": classification})
    }

"""