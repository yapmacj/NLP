import os

# Model kombinasyonları
preprocessing_types = ['lemmatized', 'stemmed']
model_types = {'cbow': 0, 'skipgram': 1}
window_sizes = [2, 4]
vector_sizes = [100, 300]

# Çalışma klasörü (model dosyalarının olduğu yer)
model_folder = './word2vec_models'

# Eğer klasör yoksa oluştur
os.makedirs(model_folder, exist_ok=True)

# Açıklamaları üret
for prep in preprocessing_types:
    for model_type, sg_value in model_types.items():
        for window in window_sizes:
            for vector_size in vector_sizes:
                model_name = f"{prep}_{model_type}_window{window}_dim{vector_size}"
                description = f"""Model: {model_name}
Preprocessing: {prep.capitalize()}
Training Type: {"CBOW" if model_type == "cbow" else "Skip-Gram"}
Window Size: {window}
Vector Size: {vector_size}
Minimum Word Frequency (min_count): 2
SkipGram (sg): {sg_value}
"""
                # .txt dosyasını oluştur
                with open(f"{model_folder}/{model_name}.txt", "w", encoding="utf-8") as f:
                    f.write(description)

print("✅ Tüm açıklama dosyaları başarıyla oluşturuldu!")
