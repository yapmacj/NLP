import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# İlk seferde nltk veri setlerini indiriyoruz
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# CSV dosyasını oku
df = pd.read_csv('reddit_soccer_comments.csv')

# Stopwords ve lemmatizer tanımlamaları
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing işlemi
def preprocess_text(text):
    # Küçük harfe çevir
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize (kelimelere ayır)
    tokens = nltk.word_tokenize(text)
    # Stopwords çıkar, Lemmatization yap
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Geri birleştir
    return ' '.join(tokens)

# Yorumları temizleyelim
df['clean_comment'] = df['comment'].astype(str).apply(preprocess_text)

# Temizlenmiş veriyi yeni dosyaya kaydedelim
df[['clean_comment']].to_csv('reddit_soccer_clean_comments.csv', index=False, encoding='utf-8')

print("✅ Preprocessing tamamlandı! reddit_soccer_clean_comments.csv dosyası oluşturuldu!")
