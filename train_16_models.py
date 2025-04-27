import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.models import Word2Vec
import re

# Gerekli NLTK verilerini indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# CSV dosyasını oku
df = pd.read_csv('reddit_soccer_comments.csv')
df = df.dropna()

# Stopwords ve diğer araçlar
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Preprocessing fonksiyonları
def preprocess_lemmatized(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def preprocess_stemmed(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return tokens

# Hazırla: lemmatized ve stemmed cümleler
lemmatized_sentences = df['comment'].astype(str).apply(preprocess_lemmatized).tolist()
stemmed_sentences = df['comment'].astype(str).apply(preprocess_stemmed).tolist()

# Model parametre kombinasyonları
preprocessing_types = {
    'lemmatized': lemmatized_sentences,
    'stemmed': stemmed_sentences
}

model_types = {
    'cbow': 0,
    'skipgram': 1
}

window_sizes = [2, 4]
vector_sizes = [100, 300]

# Eğitim ve Kaydetme
for prep_name, sentences in preprocessing_types.items():
    for model_name, sg_value in model_types.items():
        for window_size in window_sizes:
            for vector_size in vector_sizes:
                print(f"🚀 Eğitim Başlıyor: {prep_name}, {model_name}, window={window_size}, vector_size={vector_size}")

                model = Word2Vec(
                    sentences,
                    vector_size=vector_size,
                    window=window_size,
                    sg=sg_value,
                    min_count=2
                )

                model_filename = f"{prep_name}_{model_name}_window{window_size}_dim{vector_size}.model"
                model.save(model_filename)

                print(f"✅ Model kaydedildi: {model_filename}")
