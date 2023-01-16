import boto3
from clasificador import Clasificador
import logging


# Logging Definition
logging.basicConfig()
logger = logging.getLogger(__name__)

logger.setLevel(getattr(logging, 'INFO'))
logger.info('Loading Lambda Function {}'.format(__name__))

def lambda_handler(event, context):
    logger.info("Comenzando a procesar los registros")
    try:
        stopwords = get_stopwords()
        clasificador = init_clasificador()
        table = get_table_dynamodb()
        for record in event['Records']:
            if record['eventName'] == 'INSERT':
                logger.info(event)
                handler_insert(record, stopwords, clasificador, table)

    except Exception as e:
        print(e)
        return "Error insertando!"+ str(e)

def init_clasificador():
    # levanta el pickle 
    clasificador = Clasificador()
    return clasificador

def get_stopwords():
    with open('stopwords-es.txt','r',encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return stopwords

def get_nube(clean_text, stopwords):
    nube = [word for word in clean_text.split() if word not in stopwords]
    nube = " ".join(nube).lstrip().rstrip()
    return nube

def get_table_dynamodb():
    """with open('config.json') as file:
        config = json.load(file)
    aws_access_key = config['aws_access_key']
    aws_secret_access_key = config['aws_secret_access_key']
    """

    #session = boto3.Session(region_name='us-east-1',aws_access_key_id=aws_access_key,aws_secret_access_key=aws_secret_access_key)
    session = boto3.Session(region_name='us-east-1')
    ddb = session.resource('dynamodb')
    table = ddb.Table('tweet_streaming')
    return table

def get_datos_registro(record):
    #Obtengo los datos del registro
    new_image = record['dynamodb']['NewImage']
    content = {}
    for k,v in new_image.items():
        if 'S' in v:
            content[k] = v['S']
        if 'BOOL' in v:
            content[k] = v['BOOL']
        if 'L' in v:
            content[k] = v['L']
        if 'N' in v:
            content[k] = int(v['N'])
    return content

def handler_insert(record, stopwords, clasificador, table):
    print('Analizando un nuevo registro')
    # Obtengo datos del registro
    content = get_datos_registro(record)

    try:
        # limpio el texto
        clean_text = clasificador.limpiar_texto(content['text'])
        # obtengo nube y clasifico el texto
        content['nube'] = get_nube(clean_text, stopwords)
        #Clasifico el texto
        logger.info("Clasificando registro")
        content['label'] = clasificador.clasificar(clean_text)
        logger.info(content['text'], "===> ",str(content['label']))
        
        print("Insertando registro: ")
        
        response = table.put_item(Item=content)
        print("Status Code: ", str(response.status_code))
        print("Response:")
        print(response)

    except Exception as e:
        print('Error ',str(e))

    return response
