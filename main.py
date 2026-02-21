from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="ML & LLM API")

# Глобальные переменные для ленивой загрузки
sentiment_analyzer = None
llm_pipeline = None


# Схемы данных (Pydantic)
class TextRequest(BaseModel):
    text: str


class LLMRequest(BaseModel):
    prompt: str
    max_tokens: int = 50


@app.get("/")
def read_root():
    return {"message": "API is running. Go to /docs to see Swagger."}


@app.post("/predict/sentiment")
def predict_sentiment(request: TextRequest):
    global sentiment_analyzer
    if sentiment_analyzer is None:
        # Загружаем модель только при первом вызове
        sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

    result = sentiment_analyzer(request.text)[0]
    return {"text": request.text, "label": result["label"], "score": float(result["score"])}


@app.post("/predict/llm")
def predict_llm(request: LLMRequest):
    global llm_pipeline
    if llm_pipeline is None:
        # Загружаем легковесную LLM
        llm_pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

    formatted_prompt = f"<|system|>\nТы полезный ассистент. Отвечай кратко на русском.</s>\n<|user|>\n{request.prompt}</s>\n<|assistant|>\n"

    result = llm_pipeline(formatted_prompt, max_new_tokens=request.max_tokens, temperature=0.7)
    generated_text = result[0]['generated_text'].split("<|assistant|>\n")[-1]

    return {"prompt": request.prompt, "response": generated_text.strip()}