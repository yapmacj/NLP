def zipf_plot(texts, title):
    import matplotlib.pyplot as plt
    from collections import Counter
    import nltk
    import re

    all_words = []
    for text in texts:
        tokens = nltk.word_tokenize(str(text).lower())
        tokens = [re.sub(r"[^\w\s]", "", t) for t in tokens if t.isalpha()]
        all_words.extend(tokens)

    freq_dist = Counter(all_words)
    sorted_freq = sorted(freq_dist.values(), reverse=True)

    ranks = range(1, len(sorted_freq) + 1)
    frequencies = sorted_freq

    plt.figure(figsize=(8, 5))
    plt.loglog(ranks, frequencies)
    plt.title(f"Zipf Plot - {title}")
    plt.xlabel("Kelime Sırası (log)")
    plt.ylabel("Frekans (log)")
    plt.grid(True)
    plt.tight_layout()

    filename = f"zipf_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Grafik kaydedildi: {filename}")

    plt.show()
