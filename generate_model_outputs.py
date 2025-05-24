import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt')

# Giriş cümlesi (temizlenmiş haliyle)
input_sentence = "feck bet penalti favor real madrid"
input_tokens = word_tokenize(input_sentence)

# Veri setini yükle (temizlenmiş yorumlar)
df = pd.read_csv("reddit_soccer_clean_comments.csv")
df = df.dropna(subset=['clean_comment'])

# Model klasörünü belirt
model_dir = "word2vec_models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".model")]

# Yorum vektörünü oluştur
def get_vector(tokens, model):
    vecs = []
    for token in tokens:
        if token in model.wv:
            vecs.append(model.wv[token])
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)

# Çıktıları biriktir
output_rows = []

for model_file in model_files:
    print(f"\nProcessing model: {model_file}")
    model_path = os.path.join(model_dir, model_file)
    model = Word2Vec.load(model_path)

    input_vec = get_vector(input_tokens, model)
    if np.all(input_vec == 0):
        print(f"Warning: Input vector is all zeros in model {model_file}")
        continue

    similarities = []
    for comment in df['clean_comment']:
        tokens = word_tokenize(comment)
        comment_vec = get_vector(tokens, model)
        if np.all(comment_vec == 0):
            sim = 0
        else:
            sim = cosine_similarity([input_vec], [comment_vec])[0][0]
        similarities.append(sim)

    df['similarity'] = similarities
    top5 = df.sort_values(by='similarity', ascending=False).head(5)

    for i, row in enumerate(top5.itertuples(), start=1):
        output_rows.append({
            "model_name": model_file.replace(".model", ""),
            "suggestion_rank": i,
            "suggestion_text": row.clean_comment,
            "similarity": row.similarity
        })

# Sonuçları CSV olarak kaydet
output_df = pd.DataFrame(output_rows)
output_df.to_csv("model_outputs.csv", index=False)
print("\n✅ model_outputs.csv dosyası başarıyla oluşturuldu!")
