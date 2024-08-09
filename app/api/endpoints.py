from fastapi import APIRouter, HTTPException
from app.utils.similarity_calculator import calculate_similarity
from app.utils.preprocessing import preprocess_data, preprocess_text, train_model, text_to_bow
from app.models.predict import SimilarityResponse, SimilarityRequest
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
from sklearn.metrics import accuracy_score
from pathlib import Path
import pickle
import os
from tqdm import tqdm
import time


logging.basicConfig(level=logging.INFO)
router = APIRouter()


def load_data():
    file_path = "app/data/dataset.csv"
    global_dataset = pd.read_csv(file_path)
    return global_dataset


def evaluate_model(classifier, X_test, y_test):
    X_test_bow = [text_to_bow(text) for text in X_test]
    y_pred = [classifier.classify(instance) for instance in X_test_bow]
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(accuracy)
    return accuracy


@router.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity_endpoint(request: SimilarityRequest):
    similarity = calculate_similarity(request.method, request.line1, request.line2)
    return {"method": request.method, "line1": request.line1, "line2": request.line2, "similarity": similarity}


class ModelTrainer:
    def __init__(self):
        self.model_folder = "app/trained_models"
        self.model = None
        self.training_in_progress = False
        self.model_trained = False
        self.dataset = None
        self.load_model()

    def train_model(self, X_train, y_train, progress_callback=None):
        self.model = train_model(X_train, y_train)
        if progress_callback:
            progress_callback()

    def load_model(self):
        model_path = os.path.join(self.model_folder, "naive_bayes_classifier.pickle")
        if os.path.exists(model_path):
            with open(model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
                self.model_trained = True
                logging.info("model loaded from exist")


model_trainer = ModelTrainer()


# Endpoint для обучения модели
@router.get("/train_model/")
def retrain_model():
    logging.info("Start")
    if model_trainer.training_in_progress == True:
        raise HTTPException(status_code=400, detail="Model is training now")
    model_trainer.dataset = load_data()  # Установка датасета
    logging.info("data is loaded")
    model_trainer.training_in_progress = True
    start_time = time.time()
    X, y = preprocess_data(model_trainer.dataset["review"][:10000], model_trainer.dataset["sentiment"][:10000])
    end_time = time.time()
    logging.info(f"data is preprocessed, time: {end_time - start_time} (last time: 12.967535257339478)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("model is starts to train")
    with tqdm(total=len(X_train)) as pbar:
        model_trainer.train_model(X_train, y_train, progress_callback=lambda: pbar.update(1))
    logging.info("training is finished")
    accuracy = evaluate_model(model_trainer.model, X_test, y_test)
    model_folder = Path(model_trainer.model_folder)
    if model_folder.exists() and model_folder.is_dir():
        model_filename = model_folder / "naive_bayes_classifier.pickle"
        with open(model_filename, "wb") as f:
            pickle.dump(model_trainer.model, f)
            logging.info("Trained model saved successfully")
    else:
        logging.error("Model folder does not exist")
    model_trainer.training_in_progress = False
    model_trainer.model_trained = True
    return {"accuracy": accuracy}


@router.post("/predict/")
def predict_class(text: str):
    logging.info(model_trainer.model_trained)
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    if model_trainer.training_in_progress == True:
        raise HTTPException(status_code=400, detail="Model is training now")
    if model_trainer.model_trained == False:
        raise HTTPException(status_code=400, detail="Model is not trained")
    preprocessed_text = preprocess_text(text)
    new_instance_bow = text_to_bow(preprocessed_text)  # Replace this with your actual preprocessing function
    predicted_class = model_trainer.model.classify(new_instance_bow)
    return {"predicted_class": predicted_class, "preprocessed_text": preprocessed_text}
