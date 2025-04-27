import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# NLTK download (gerekirse)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Model ve verileri yükleyelim
model = Word2Vec.load("word2vec_reddit_comments.model")
df = pd.read_csv("reddit_soccer_clean_comments.csv")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing fonksiyonu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# Yorumu vektör ortalamasına çeviren fonksiyon
def get_comment_vector(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv.key_to_index:  # Token modelde varsa
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Kullanıcıdan bir yorum alalım
user_comment = input("🎯 Yorumunuzu girin: ")
user_tokens = preprocess_text(user_comment)
user_vector = get_comment_vector(user_tokens, model)

# Veri setindeki tüm yorumlar için vektörleri çıkaralım
all_vectors = []
for comment in df['clean_comment']:
    tokens = preprocess_text(str(comment))
    vector = get_comment_vector(tokens, model)
    all_vectors.append(vector)

# Cosine Similarity hesaplayalım
similarities = cosine_similarity([user_vector], all_vectors)

# En yüksek benzerliğe sahip 5 yorumu bulalım
top_indices = similarities.argsort()[0][-5:][::-1]  # En büyükten küçüğe

print("\n✅ En Benzer 5 Yorum:")
for idx in top_indices:
    print(f"\nBenzerlik Skoru: {similarities[0][idx]:.4f}")
    print(f"Yorum: {df.iloc[idx]['clean_comment']}")
