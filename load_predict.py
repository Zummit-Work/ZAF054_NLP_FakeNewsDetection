import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from joblib import Parallel, delayed
import joblib
model = joblib.load('model.h5')
import pandas as pd
import numpy as np
df = pd.read_csv('C:/Users/asha/Desktop/zummit/fake news/train.csv')
from nltk.stem.porter import PorterStemmer
port_stem = PorterStemmer()
import re
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

df['content'] = df['content'].apply(stemming)
X = df['content'].values
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
pred=model.predict(X)