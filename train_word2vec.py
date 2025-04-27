import pandas as pd
import nltk
from gensim.models import Word2Vec

# NLTK tokenizasyon için lazım olabilir
nltk.download('punkt')

# Temizlenmiş yorumları oku
df = pd.read_csv('reddit_soccer_clean_comments.csv')

# NaN değerleri kaldır
df = df.dropna()

# Her yorumu tokenize edelim
sentences = df['clean_comment'].apply(nltk.word_tokenize).tolist()

# Word2Vec modelini eğitelim
model = Word2Vec(
    sentences,
    vector_size=100,    # Her kelimeyi 100 boyutlu bir vektörle temsil ediyoruz
    window=5,           # Bir kelimenin çevresindeki 5 kelimeye bakıyor
    min_count=2,        # En az 2 kere geçen kelimeleri dikkate al
    sg=1                # sg=1 -> Skip-Gram; (0 olursa CBOW kullanılır)
)

# Eğitimi tamamlayınca modeli kaydedelim
model.save("word2vec_reddit_comments.model")

print("✅ Word2Vec eğitimi tamamlandı ve model word2vec_reddit_comments.model olarak kaydedildi!")
