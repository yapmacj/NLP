import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Gerekli NLTK verileri
nltk.download('punkt')
nltk.download('stopwords')

# Stemmer ve stopword listesi
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Stem ile ön işleme fonksiyonu
def stem_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# CSV'den temizlenmiş (lemmatized) yorumları oku
df = pd.read_csv("reddit_soccer_clean_comments.csv")
texts = df["clean_comment"].dropna().astype(str).apply(stem_preprocess).tolist()

# TF-IDF hesapla
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

# DataFrame'e dönüştür
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df.to_csv("tfidf_stemmed.csv", index=False, encoding="utf-8-sig")

print("✅ tfidf_stemmed.csv başarıyla oluşturuldu.")
