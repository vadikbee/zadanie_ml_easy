from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running. Go to /docs to see Swagger."}

def test_predict_sentiment_positive():
    response = client.post(
        "/predict/sentiment",
        json={"text": "Я очень люблю программировать, это потрясающе!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert "score" in data

def test_predict_sentiment_negative():
    response = client.post(
        "/predict/sentiment",
        json={"text": "Ужасный день, всё сломалось и ничего не работает."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"
    assert "score" in data