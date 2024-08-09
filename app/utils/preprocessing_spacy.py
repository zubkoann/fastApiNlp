import nltk
from nltk.corpus import words
import spacy
from spacy.tokens import Doc
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.classify import NaiveBayesClassifier
from sklearn.pipeline import Pipeline
import logging
from collections import Counter
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
nlp = spacy.load("en_core_web_lg")

pipeline = {
    "remove_html": True,
    "remove_url": True,
    "tokenize": False,
    "remove_stopwords": True,
    "lemmatize": True,
    "lowercasing": True,
    "remove_punctuation": True,
    "remove_extra_ws": True,
    "remove_frequent_words": False,
    "spelling_correction": False,
}


def preprocess_text(
    text,
    pipeline=pipeline,
):
    if not text:
        return ""

    stop_words = set(nlp.Defaults.stop_words)

    if pipeline.get("remove_html", False):
        text = re.sub(r"<.*?>", "", text)

    if pipeline.get("remove_url", False):
        text = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "", text)

    if pipeline.get("lowercasing", False):
        text = text.lower()

    doc = nlp(text)

    if pipeline.get("remove_stopwords", False):
        tokens = [token for token in doc if token.text.lower() not in stop_words]
    else:
        tokens = [token for token in doc]

    if pipeline.get("remove_punctuation", False):
        tokens = [token for token in tokens if not token.is_punct]

    if pipeline.get("lemmatize", False):
        tokens = [token.lemma_ for token in tokens]

    if pipeline.get("remove_extra_ws", False):
        tokens = [token.strip() for token in tokens if token.strip()]

    if pipeline.get("spelling_correction", False):
        words_list = words.words()
        corrected_tokens = []
        for token in tokens:
            corrected_token = min(words_list, key=lambda x: nltk.edit_distance(x, token))
            corrected_tokens.append(corrected_token)
        tokens = corrected_tokens

    if pipeline.get("remove_frequent_words", False):
        fdist = Counter(tokens)
        tokens = [token for token in tokens if fdist[token] < fdist.most_common(1)[0][1]]

    if pipeline.get("tokenize", False):
        resp = tokens
    else:
        resp = " ".join(tokens)

    return resp


def preprocess_data(X, y):
    preprocessed_X = []
    with tqdm(total=len(X), desc="Preprocessing") as pbar:
        for text in X:
            preprocessed_text = preprocess_text(text)
            preprocessed_X.append(preprocessed_text)
            pbar.update(1)
    return preprocessed_X, y


def train_model(X, y):
    logging.info(len(X))
    logging.info(len(y))

    X_bow = [text_to_bow(text) for text in X]
    labeled_data = list(zip(X_bow, y))
    classifier = nltk.classify.NaiveBayesClassifier.train(labeled_data)
    return classifier


def text_to_bow(text):
    words = text.split()
    bow = Counter(words)
    return bow
