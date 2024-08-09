import re
import emoji
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)

model_dir = "./app/trained_models/transformers/model"
tokenizer_dir = "./app/trained_models/transformers/tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)


def add_space_around_emojis(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"\s+(:\w+:)\s+", r" \1 ", text)
    text = re.sub(r"(?<! )(:\w+:)(?! )", r" \1 ", text)
    return text


def remove_url(text):
    pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return cleaned_text


def split_sentences(text):
    sentence_endings = re.compile(r"(?<=[.!?])\s*(?=[A-Za-zА-Яа-яЁёЇїІіЄєҐґ])")
    sentences = sentence_endings.split(text)
    return [sentence.strip() for sentence in sentences]


def clean_text(text):
    text = add_space_around_emojis(text)
    text = remove_url(text)
    text = " ".join(text.split("\\n"))
    text = re.sub(r"(?<![\s!])!", " !", text)
    text = re.sub(r"(?<![\s\?])\?", " ?", text)
    text = re.sub(r"[\r\n\v\f_,)]+", " ", text)
    text = re.sub(r"\b(?!5\b|10\b|100\b)\d+\b", " ", text)
    text = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁіІїЇєЄҐґ.!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    cleaned_text = text.strip().lower()
    return cleaned_text


def process_text(text, flag_filter=True, flag_split=True):
    text = clean_text(text)
    if flag_split == True and flag_filter == True:
        sentences = split_sentences(text)
        filtered_sentences = [sentence for sentence in sentences if "KEY" in sentence]
        return filtered_sentences
    if flag_split == True and flag_filter == False:
        sentences = split_sentences(text)
        return sentences
    return text


def process_dataframe_for_check_by_text(text, flag_filter, flag_split):
    processed_text = process_text(text, flag_filter, flag_split)
    return processed_text


def get_sentiment(text, return_type="label"):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
    if return_type == "label":
        return model.config.id2label[proba.argmax()]
    return proba


def get_overall_sentiment(text):
    parts = process_dataframe_for_check_by_text(text, flag_filter=False, flag_split=True)

    logging.info(parts)
    sentiments = []
    if len(parts) > 0:
        for part in parts:
            sentiment = get_sentiment(part, return_type="proba")
            logging.info(sentiment)
            sentiments.append(sentiment)
        mean_sentiment = np.max(sentiments, axis=0)
        logging.info(mean_sentiment)
        if mean_sentiment[1] > 0.81:
            label = "toxic"
        else:
            label = "neutral"
        return label, mean_sentiment[1], mean_sentiment, parts, sentiments
    else:
        return "neutral", mean_sentiment[1], mean_sentiment, parts, sentiments
