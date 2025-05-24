import pandas as pd

# Gerekli dosyaları yükle
df_similarity = pd.read_csv("model_average_similarity_scores.csv")
df_manual = pd.read_csv("manual_scoring_filled.csv")

# Manuel skorları model bazında grupla
manual_avg = df_manual.groupby("model_name")["your_score_1_to_5"].mean().reset_index()
manual_avg = manual_avg.rename(columns={"your_score_1_to_5": "average_manual_score"})

# Cosine similarity dosyasıyla birleştir
combined = pd.merge(df_similarity, manual_avg, on="model_name")
combined = combined.rename(columns={"similarity": "average_cosine_similarity"})

# En yüksek puana göre sırala
combined = combined.sort_values(by=["average_manual_score", "average_cosine_similarity"], ascending=False)

# CSV'ye kaydet
combined.to_csv("model_comparison_scores.csv", index=False)

print("✅ Model karşılaştırma tablosu model_comparison_scores.csv dosyasına kaydedildi.")
