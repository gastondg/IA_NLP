import pickle
from preprocessor import Processor


class Clasificador():

    def __init__(self):
        clasificador = 'clasificador_multinomial_v2.pkl'
        vectorizador = 'vectorizador_tfidf_v2.pkl'

        self.P = Processor()
        with open(clasificador, 'rb') as m1:
            self.clasif = pickle.load(m1,encoding='latin1')
    
        with open(vectorizador, 'rb') as m2:
            self.vect = pickle.load(m2,encoding='latin1')

    def limpiar_texto(self, text):
        """ 
        Recibe texto sucio y lo limpia
        """
        return self.P.process(text)

    def clasificar(self, texto):
        """ 
        Devuelve la clasificacion del texto
        """
        #texto = self.limpiar_texto(texto)
        clasificacion = self.clasif.predict(self.vect.transform([texto]))[0]
        return clasificacion

