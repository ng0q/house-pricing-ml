import joblib
from pathlib import Path

MODEL_DIR = Path('model')

model = None
target_encode = None

def load_artifacts() -> None:
    global model, target_encoder

    model_path = MODEL_DIR / 'model.pkl'
    encoder_path = MODEL_DIR / 'target_encoding.pkl'

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path.resolve()}. "
            f"Start `python -m src.train` before."
        )
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"TargetEncoder not found: {encoder_path.resolve()}. "
            f"Start `python -m src.train` before."
        )
    model = joblib.load(model_path)
    target_encoder = joblib.load(encoder_path)

    print(f"Model loaded from {model_path.resolve()}")
    print(f"TargetEncoder loaded from{encoder_path.resolve()}")