import boto3


    
def datos_registro(record):
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