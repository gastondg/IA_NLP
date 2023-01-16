import spacy
import re
import string

class Processor:
    """
    Una clase hecha para limpiar el texto que traen los tweets.
    
    El metodo process limpia el texto haciendo uso de los demas metodos
    """
    # cargar el modelo es_core_news_sm 
    
    def __init__(self):
        self.nlp = spacy.load("es_core_news_sm")
        self.dirtyReps = re.compile(r'([^lL0])\1{1,}')
        self.dirtySpaces = re.compile(r'(\.|,|:|;|!|\?|\[|\]|\(|\))[A-Za-z0-9]+')
        self.dirtyK = re.compile('[^o]k')
        self.dirtyJaja = re.compile(r'[ja]{5,}')
        self.dirtyJeje = re.compile(r'[je]{5,}')
        self.uglySeparator = 'THIS-IS-A-SEPARATOR'
        self.urlPattern=re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
        

    def replaceAccents(self,word):
        """ Reemplaza las tildes """
        word = word.replace('í','i')
        word = word.replace('ó','o')
        word = word.replace('ò','o')
        word = word.replace('ñ','n')
        word = word.replace('é','e')
        word = word.replace('è','e')
        word = word.replace('á','a')
        word = word.replace('à','a')
        word = word.replace('ü','u')
        word = word.replace('ú','u')
        word = word.replace('ö','o')
        word = word.replace('ë','e')
        word = word.replace('ï','i')
        return word

    def processDetails(self,word):
        """ 
        Clase 'comodin' para simplificar el procesamiento
        
        Por ejemplo: re -> muy, 100% -> muy, #hashtag -> hashtag
        
        Se puede mejorar haciendo un diccionario de reemplazo
        word es toda la frase> word = "esto es una 100% frase" -> "esto es una muy frase"
        """
        word = word.replace('#', '')
        word = word.replace(' re ',' muy ')
        word = word.replace(' 100% ', ' muy ')
        word = word.replace('mercado libre', ''.join(['mercado','libre']))
        word = word.replace('mercado pago', ''.join(['mercado','pago']))
        word = word.replace(' x ',' por ')
        word = word.replace(' q ', ' que ')
        word = word.replace(' qu ', ' que ')
        word = word.replace(' qeh ', ' que ')
        word = word.replace(' qhe ', ' que ')
        word = word.replace(' qe ', ' que ')
        word = word.replace(' ke ', ' que ')
        word = word.replace(' keh ', ' que ')
        word = word.replace(' khe ', ' que ')
        word = word.replace(' k ', ' que ')
        word = word.replace('lll','ll')
        word = word.replace('la puta madre','lpm')
        word = word.replace('la puta que te pario','lpqtp')
        word = word.replace('la concha de la lora','lcdll')
        word = word.replace('hijo de puta', 'hdp')
        
        return word

    def processJaja(self,word):
        """
        jajajaj, jajja, jajaj, jejejaja, jejej, etc -> jaja
        """
        while self.dirtyJaja.search(word)!=None:
            word = word.replace(self.dirtyJaja.search(word).group(),'jaja')
        while self.dirtyJeje.search(word)!=None:
            word = word.replace(self.dirtyJeje.search(word).group(),'jaja')
        return word    
    
    def processRep(self,word):
        """
        Saca las repeticiones por ej: buenasss, bueeena, essss -> buenas, buena, es
        No remueve la doble l
        """
        while self.dirtyReps.search(word)!=None:
            word = word.replace(self.dirtyReps.search(word).group(),
                                self.dirtyReps.search(word).group()[0])
        return word

    def processSpaces(self,word):
        """
        Remueve espacios de sobra, incluidos los de adelante de la sentencia y los del final
        """
        while self.dirtySpaces.search(word)!=None:
            word = word.replace(self.dirtySpaces.search(word).group()[0],
                                self.dirtySpaces.search(word).group()[0]+' ')
        word = word.lstrip().rstrip()
        return word
    
    def processK(self,word):
        """
        Remueve la k en palabras como kilombo -> quilombo
        """
        while self.dirtyK.search(word)!=None:
            word = word.replace(self.dirtyK.search(word).group(),
                                self.dirtyK.search(word).group()[0]+'qu')
        return word
    
    def processNumbers(self,word):
        """
        Los numero normalmente no tienen peso en una frase, los remueve
        Excepto por la frase "esta todo de 10" -> "esta todo de diez"
        """
        word = word.replace('10',' diez')
        word = re.sub(r'\d+', '', word)  
        return word
    
    def removePuntuation(self, word):
        """
        Remueve todas las puntuaciones
        """
        word = "".join([char for char in word if char not in string.punctuation]) 
        return word
    
    def removeLinks(self, word):
        """
        Elimina los links y deja solo 'url' como constancia de un link eliminado
        Puede servir para encontrar noticias neutras
        """
        return self.urlPattern.sub(' url ', word)
    
    def removeUsers(self, word):
        """
        Elimina los users para no sesgar la clasificacion
        """
        return re.sub('@[^\s]+','',word)

    def process(self,x,lematizar=True):
        
        if len(x) > 0:
            str(x).replace('\r','').replace('\n','')
            x = self.removeLinks(x)
            x = self.removeUsers(x)
            x = self.replaceAccents(x.lower())  
            x = self.removePuntuation(x)
            x = self.processNumbers(x)
            x = self.processRep(x)
            x = self.processJaja(x)
            x = self.processSpaces(x)
            x = self.processDetails(x)
            x = self.processK(x)
            
            if lematizar:
                x = self.nlp(x)
                x = " ".join([token.lemma_ for token in x])
        
        return x


if __name__ == '__main__':
    limpieza = Processor()
    estring = "Haber si esta funcionaria www.github.com ESTAMOSSSS PROBando"
    print(limpieza.process(estring))
    print(limpieza.process(estring, lematizar=False))