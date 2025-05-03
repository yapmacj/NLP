# Metin Tabanlı Veri Setleri ile Yapay Zeka Modelleri Geliştirme

Bu projede, Reddit platformundan alınan yorumlar üzerinde doğal dil işleme (NLP) teknikleri uygulanarak TF-IDF ve Word2Vec gibi vektörleştirme yöntemleriyle metin madenciliği yapılmıştır.

## 🔍 Proje İçeriği

- **Veri Kaynağı:** Reddit API (`r/soccer` subreddit)
- **Veri Dosyası:** `reddit_soccer_comments.csv`
- **Temizlenmiş Veri:** `reddit_soccer_clean_comments.csv` (lemmatized)
- **Ön İşleme:** Stopword temizliği, lowercasing, tokenization, lemmatization, stemming
- **Zipf Analizi:** Ham, lemmatized ve stemmed veriler için log-log grafikleri
- **TF-IDF:** `tfidf_lemmatized.csv` ve `tfidf_stemmed.csv`
- **Word2Vec:** Toplam 16 model üretildi (`CBOW` & `SkipGram`, window=2/4, dim=100/300, lemma/stem)
- **Model Açıklamaları:** `generate_model_descriptions.py` ile her model için açıklama dosyası

## 📁 Klasör Yapısı

```
.
├── reddit_soccer_comments.csv
├── reddit_soccer_clean_comments.csv
├── tfidf_lemmatized.csv
├── tfidf_stemmed.csv
├── zipf_ham_veri.png
├── zipf_lemmatized_veri.png
├── zipf_stemmed_veri.png
├── word2vec_models/
│   ├── lemmatized_cbow_window2_dim100.model
│   ├── ...
├── model_descriptions/
│   ├── lemmatized_cbow_window2_dim100.txt
│   ├── ...
├── scripts/
│   ├── veri_cek.py
│   ├── preprocess.py
│   ├── train_16_models.py
│   ├── generate_model_descriptions.py
│   ├── recommend.py
```

## ⚙ Kullanılan Kütüphaneler

- `nltk`
- `pandas`
- `gensim`
- `matplotlib`
- `sklearn`
- `praw`

## 📌 Örnek Word2Vec Çıktısı

```
Model: lemmatized_cbow_window2_dim100.model
Kelime: goal
Benzer Kelimeler:
  4        (0.9919)
  month    (0.9918)
  minute   (0.9917)
  10       (0.9916)
  two      (0.9915)
```

## 📄 Rapor
