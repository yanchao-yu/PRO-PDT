from recognition.config import SIMILARITY_THRESHOLD, LOW_THRESHOLD
from recognition.core.inference.embedding_service import EmbeddingService
from recognition.core.inference.enrollment_service import enroll_user
from recognition.core.inference.model_service import ModelWeightService
from recognition.core.utils.database import Database, IdentityClass
from datetime import datetime

def identify_user(image_path, model_name, model_path, gallery: list[IdentityClass], threshold=SIMILARITY_THRESHOLD):
    print("identifying user....")
    model = ModelWeightService.load(model_name=model_name, weight_path=model_path)
    embedding = EmbeddingService.get_embedding(image_path, model)
    database = Database()
    low_threshold = LOW_THRESHOLD  # threshold for low confidence matches due to image is new
    if embedding is None:
        return {
            "name": "Face not detected",
            "role": None,
            "score": 0.0
        }

    best_match, best_score = database.check_similarity(gallery, embedding)

    # If no match is found or the best score is below the threshold, enroll the user
    if best_score <= low_threshold:
        print("Low confidence match, enrolling user as unknown...")
        unknown_user = f"unknown_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        enroll_user(
            name=unknown_user,
            role="unknown",
            folder=image_path,
            model_name=model_name,
            weight_path=model_path,
            database=database
        )

    if best_match is not None and best_score > threshold:
        return {
            "name": best_match.name,
            "role": best_match.role,
            "score": round(best_score, 4),
        }
    else:
        return {"name": "unknown", "role": None, "score": round(best_score, 4)}
