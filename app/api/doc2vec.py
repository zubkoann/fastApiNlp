import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pydantic import BaseModel
from fastapi import APIRouter
import logging
import os
from app.utils.preprocessing import preprocess_text
from sklearn.cluster import DBSCAN
from nltk.tokenize import word_tokenize
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
doc2vec_router = APIRouter()


def load_data():
    file_path = "app/data/bbc_data.csv"
    global_dataset = pd.read_csv(file_path)
    global_dataset.dropna(subset=["data"])
    global_dataset["processed"] = global_dataset["data"].apply(preprocess_text)
    return global_dataset


def classify_sentences(model, sentences, topn=1):
    results = []
    for sentence in sentences:
        vector = model.infer_vector(word_tokenize(sentence.lower()))
        similar_docs = model.dv.most_similar([vector], topn=topn)
        results.append(similar_docs[0][0])
    return results


class ModelTrainer:
    def __init__(self):
        self.model_folder = "app/trained_models"
        self.model = None
        self.training_in_progress = False
        self.model_trained = False
        self.dataset = None
        self.load_model()

    def load_model(self):
        model_path = os.path.join(self.model_folder, "doc2vec_model.d2v")
        if os.path.exists(model_path):
            with open(model_path, "rb") as model_file:
                self.model = Doc2Vec.load(model_path)
                logging.info("model loaded from exist")


doc2vec_model_trainer = ModelTrainer()


class ClassifyRequest(BaseModel):
    sentences: List[str]

    class Config:
        schema_extra = {
            "example": {
                "sentences": [
                    "Soul sensation ready for awards  South West teenage singing sensation,",
                    "Top gig award for Scissor Sisters  New York band Scissor Sisters have won a gig of the year award",
                    "The Producers scoops stage awards  The Producers has beaten Mary Poppins in the battle of the blockbuster West End musicals at the Olivier Awards. ",
                    "Japan bank shares up on link talk  Shares of Sumitomo Mitsui Financial (SMFG), and Daiwa Securities jumped amid speculation that two of Japans biggest financial companies will merge.  Financial newspaper Nihon Keizai Shimbun claimed that the firms will join up next year and already have held discussions with Japanese regulators. The firms denied that they are about to link up, but said they are examining ways of working more closely together. SMFG shares climbed by 2.7% to 717,000, and Daiwa added 5.3% to 740 yen.  ",
                ]
            }
        }


@doc2vec_router.get("/doc2vec/train_model/")
def train_model():
    doc2vec_model_trainer.dataset = load_data()  # Установка датасета
    tagged_data = [
        TaggedDocument(words=word_tokenize(row["processed"]), tags=[str(row["labels"])])
        for _, row in doc2vec_model_trainer.dataset.iterrows()
    ]
    model = Doc2Vec(vector_size=20, min_count=1, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model_path = os.path.join(doc2vec_model_trainer.model_folder, "doc2vec_model.d2v")
    model.save(model_path)
    return {"detail": "Model trained and saved successfully"}


@doc2vec_router.post("/doc2vec/group_sentences/")
def group_sentences(request: ClassifyRequest) -> Dict[str, List[str]]:
    sentences = request.sentences
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
    predicted_topics = classify_sentences(doc2vec_model_trainer.model, preprocessed_sentences, topn=2)
    grouped_sentences = {}
    for sentence, topic in zip(sentences, predicted_topics):
        if topic not in grouped_sentences:
            grouped_sentences[topic] = []
        grouped_sentences[topic].append(sentence)
    return grouped_sentences


def classify_sentences(model, sentences, topn=1):
    results = []
    for sentence in sentences:
        vector = model.infer_vector(word_tokenize(sentence.lower()))
        similar_docs = model.dv.most_similar([vector], topn=topn)
        results.append(similar_docs[0][0])
    return results
