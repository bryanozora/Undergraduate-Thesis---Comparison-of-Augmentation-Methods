import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords

# ======================
# SETUP
# ======================
STOPWORDS = set(stopwords.words("indonesian"))

# ======================
# NORMALISASI SEDERHANA
# ======================
def normalize(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", " ", text)     # hapus simbol
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ======================
# MAIN
# ======================
def main():
    df = pd.read_csv("data_80.csv")

    all_words = []

    for text in df["Text"].dropna():
        text = normalize(text)
        tokens = text.split()
        tokens = [
            t for t in tokens
            if t not in STOPWORDS and len(t) > 2
        ]
        all_words.extend(tokens)

    counter = Counter(all_words)

    vocab_df = pd.DataFrame(
        counter.items(),
        columns=["kata", "frekuensi"]
    ).sort_values("frekuensi", ascending=False)

    vocab_df.to_csv("kata_kandidat_sinonim.csv", index=False)

    print("Selesai.")
    print("Jumlah kata unik non-stopword:", len(vocab_df))
    print("10 kata teratas:")
    print(vocab_df.head(10))

if __name__ == "__main__":
    main()
