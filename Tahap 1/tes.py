from nltk.corpus import wordnet as wn

def get_synonyms_id(word):
    synonyms = set()

    for syn in wn.synsets(word, lang="ind"):
        for lemma in syn.lemmas(lang="ind"):
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word:
                synonyms.add(synonym)

    return list(synonyms)

print(get_synonyms_id("mati"))