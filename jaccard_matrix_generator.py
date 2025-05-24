import pandas as pd
import numpy as np

# Model öneri dosyasını oku
df = pd.read_csv("model_outputs.csv")

# Her modelin önerdiği 5 yorumun setini al
model_sets = df.groupby("model_name")["suggestion_text"].apply(lambda x: set(x.str.lower())).to_dict()

# Tüm model isimlerini sırayla al
model_names = list(model_sets.keys())

# Jaccard skorları için boş matris oluştur
jaccard_matrix = pd.DataFrame(index=model_names, columns=model_names)

# Jaccard benzerlik hesapla
for model_a in model_names:
    for model_b in model_names:
        set_a = model_sets[model_a]
        set_b = model_sets[model_b]
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        score = intersection / union if union != 0 else 0
        jaccard_matrix.loc[model_a, model_b] = round(score, 3)

# CSV olarak kaydet
jaccard_matrix.to_csv("jaccard_matrix.csv")

print("✅ Jaccard benzerlik matrisi jaccard_matrix.csv olarak kaydedildi.")
