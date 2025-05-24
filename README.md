# Metin Tabanlı Veri Setleri ile Yapay Zeka Modelleri Geliştirme

Bu projede, Reddit platformundan alınan yorumlar üzerinde doğal dil işleme (NLP) teknikleri uygulanarak TF-IDF ve Word2Vec gibi vektörleştirme yöntemleriyle metin madenciliği gerçekleştirilmiştir. Amaç, girilen bir futbol yorumu ile anlamsal olarak benzer olan yorumları otomatik olarak bulmak ve farklı modellerin performansını karşılaştırmaktır.

---

## 🔍 Proje İçeriği

- **Veri Kaynağı:** Reddit API (`r/soccer` subreddit)
- **Veri Dosyası:** `reddit_soccer_comments.csv`
- **Temizlenmiş Veri:** `reddit_soccer_clean_comments.csv` (lemmatized & stemmed)
- **Ön İşleme:** Stopword temizliği, lowercasing, tokenization, lemmatization, stemming
- **Zipf Analizi:** Ham, lemmatized ve stemmed veriler için log-log grafikleri
- **TF-IDF:** `tfidf_lemmatized.csv` ve `tfidf_stemmed.csv`
- **Word2Vec:** Toplam 16 model (“CBOW” & “SkipGram”, window=2/4, dim=100/300, lemma/stem)
- **Model Çıktıları:** 18 model önerisi `model_outputs.csv` dosyasında saklanmıştır
- **Giriş Yorumu:** `"feck bet penalti favor real madrid"`

---

## 📁 Klasör Yapısı
```
.
├── reddit_soccer_comments.csv
├── reddit_soccer_clean_comments.csv
├── tfidf_lemmatized.csv
├── tfidf_stemmed.csv
├── zipf_*.png
├── word2vec_models/
│   ├── *.model (16 adet)
├── model_descriptions/
│   ├── *.txt
├── outputs/
│   ├── model_outputs.csv
│   ├── model_average_similarity_scores.csv
│   ├── manual_scoring_filled.csv
│   ├── model_comparison_scores.csv
│   ├── jaccard_matrix.csv
├── scripts/
│   ├── veri_cek.py
│   ├── preprocess.py
│   ├── train_16_models.py
│   ├── generate_model_descriptions.py
│   ├── recommend.py
│   ├── compare_models_scores.py
│   ├── jaccard_matrix_generator.py
│   ├── save_model_table_image.py
```

---

## ⚙ Kodların Açıklaması

### `veri_cek.py`
Reddit API kullanarak `r/soccer` subreddit'inden yorumları çeker ve `reddit_soccer_comments.csv` dosyasına kaydeder.

### `preprocess.py`
Ham verileri temizler:
- Noktalama kaldırma
- Küçük harfe dönüşürme
- Tokenization
- Stopword silme
- Lemmatization / Stemming uygular

### `train_16_models.py`
16 farklı Word2Vec modelini (CBOW vs SkipGram, dim 100/300, window 2/4, lemma/stem) eğitir ve `word2vec_models/` klasörüne kaydeder.

### `generate_model_descriptions.py`
Her `.model` dosyası için aynı isimde `.txt` formatında açıklama dosyası oluşturur.

### `recommend.py`
Kullanıcıdan bir giriş cümlesi alır. Tüm modelleri kullanarak en benzer 5 yorumu ve cosine similarity skorunu hesaplar. `model_outputs.csv` dosyasına yazar.

### `compare_models_scores.py`
`model_outputs.csv` ve `manual_scoring_filled.csv` dosyalarını birleştirir. Ortalama cosine similarity + anlamsal puan hesaplayarak `model_comparison_scores.csv` dosyasını oluşturur.

### `jaccard_matrix_generator.py`
Her modelin ilk 5 önerisini set olarak alır. 18x18 Jaccard benzerlik matrisini `jaccard_matrix.csv` dosyasına yazar.

### `save_model_table_image.py`
`model_comparison_scores.csv` tablosunu görselleştirir ve PNG olarak kaydeder.

---

## 📊 Model Karşılaştırması Özet

| Model | Avg. Cosine | Avg. Anlamsal Skor |
|-------|-------------|---------------------|
| stemmed_cbow_window2_dim300 | 0.9994 | 4.4 |
| stemmed_cbow_window2_dim100 | 0.9989 | 4.4 |
| stemmed_cbow_window4_dim100 | 0.9987 | 4.2 |

---

## 🔄 Jaccard Matris Özeti

- 18 modelin ilk 5 önerisi set olarak karşılaştırıldı
- Aynı önerileri sunan modellerin Jaccard skoru `1.0`
- Benzer olmayanlar `0.0 - 0.3` aralığında
- Matris dosyası: `jaccard_matrix.csv`

---

## 📄 Rapor

Tüm analizlerin detaylı açıklandığı PDF raporu şu dosyadadır:

📄 `final_nlp_project_report.pdf`

---

Hazırlayan: **[Senin Adın]**  
Ders: **Doğal Dil İşleme**  
Tarih: [Final Teslim Tarihi]
