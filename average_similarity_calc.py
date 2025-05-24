import pandas as pd

# Dosyayı oku (aynı dizinde olmalı)
df = pd.read_csv("model_outputs.csv")

# Her model için ortalama similarity skorunu hesapla
avg_similarities = df.groupby("model_name")["similarity"].mean().reset_index()

# Skora göre azalan sıraya koy
avg_similarities = avg_similarities.sort_values(by="similarity", ascending=False)

# Dosyaya yaz
avg_similarities.to_csv("model_average_similarity_scores.csv", index=False)

print("✅ Ortalama benzerlik skorları hesaplandı ve model_average_similarity_scores.csv dosyasına kaydedildi.")
