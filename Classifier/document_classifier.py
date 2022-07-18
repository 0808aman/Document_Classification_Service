import joblib
import re
from bs4 import BeautifulSoup

class DocumentClassifier:
    def __init__(self ):
        self.c_vect = joblib.load('/Users/amanpandey/Downloads/NLP Project/Artifacts/c_vect.pickle')
        self.tfidf = joblib.load('/Users/amanpandey/Downloads/NLP Project/Artifacts/tfidf_vect.pickle')
        self.le = joblib.load('/Users/amanpandey/Downloads/NLP Project/Artifacts/le.pickle')
        self.de_tree = joblib.load('/Users/amanpandey/Downloads/NLP Project/Artifacts/de_tree.pickle')
        self.log_reg = joblib.load('/Users/amanpandey/Downloads/NLP Project/Artifacts/log_reg.pickle')
        self.ran_for = joblib.load('/Users/amanpandey/Downloads/NLP Project/Artifacts/ran_for.pickle')
        self.svm = joblib.load('/Users/amanpandey/Downloads/NLP Project/Artifacts/svm.pickle')
        

    def extract(self, file):
        soup = BeautifulSoup(file, 'html.parser')
  
        page = soup.find('div', class_ = "ocr_page")
        if page != None:
            text = page.get_text()
        else:
            return None
        return text
    
    def text_cleaning(self, text):
        # print(text)
        
        text = re.sub(':', ',', text)
        words = text.split("\n")
        words = [word for word in words if len(word) > 2]
        text = " ".join(words)
        text = re.sub(r"[,.;:-_+={}[]()%@#?!&$/]+\ *", " ", text)
        text = re.sub(r'[^\x00-\x7F]+'," ", text)
        text = re.sub(' +', " ", text)
        text = " ".join([word for word in text.split(" ") if len(word) > 2])

        return text

    def count_vect(self, text):
        A = self.c_vect.transform([text])

        return A

    def get_tfidf_word_vectorizer(self, text):
        A = self.tfidf.transform([text])

        return A   

    
    
    
    def classify(self, text):
        a = self.extract(text)
        txt = self.text_cleaning(a)
        cv = self.count_vect(txt)
        tf = self.get_tfidf_word_vectorizer(txt)
        y_pred = self.log_reg.predict(cv)
        prob = self.log_reg.predict_proba(cv)[0][y_pred[0]]
        dict = {'Class': self.le.inverse_transform(y_pred)[0],'Prob': prob}
        #return self.le.inverse_transform(y_pred)
        return dict
        

        #return y_pred

        #pass
        # extract text from hocr
        # text clean
        # vectorize
        # predict
        # return json {"class": <>, "probability": <>}