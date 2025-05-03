import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Temizlenmiş yorumları oku
df = pd.read_csv("reddit_soccer_clean_comments.csv")
texts = df["clean_comment"].dropna().astype(str).tolist()

# TF-IDF hesapla
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

# DataFrame olarak kaydet
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df.to_csv("tfidf_lemmatized.csv", index=False, encoding="utf-8-sig")

print("✅ tfidf_lemmatized.csv dosyası oluşturuldu.")
