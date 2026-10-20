import os

import torch
import torch.nn.functional as F

from recognition.config import SIMILARITY_THRESHOLD
from recognition.core.inference.embedding_service import EmbeddingService
from recognition.core.inference.model_service import ModelWeightService
from recognition.core.utils.database import Identities, IdentityClass, Database


def enroll_user(
        name, role, folder, model_name: str, weight_path: str, database: Identities
):
    model = ModelWeightService.load(model_name=model_name, weight_path=weight_path)
    embeddings = []


    # Check if the folder exists and is a directory
    if os.path.isdir(folder):
        print(f"Enrolling user {name} with total {len(os.listdir(folder))} images")
        for filename in os.listdir(folder):
            if str(filename).split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
                path = os.path.join(folder, filename)

                embedding = EmbeddingService.get_embedding(path, model)
                if embedding is None:
                    print(f"Skipping {filename}: no face detected")
                    continue

                embeddings.append(embedding)

    else:
        print(f"Enrolling user {name} with single image {folder}")

        # If the folder is not a directory, treat it as a single image file
        embedding = EmbeddingService.get_embedding(folder, model)
        embeddings.append(embedding)

    if not embeddings:
        return f"No images found in {folder}"

    embeddings = torch.cat(embeddings, dim=0)
    embedding_sum = torch.sum(embeddings, dim=0)
    new_embedding = embeddings.mean(dim=0, keepdim=True)
    new_embedding = F.normalize(new_embedding, dim=1)

    best_match, best_score = Database().check_similarity(database.get_all(), new_embedding)

    # if user already exists on the db,
    # then update the average embedding using ERA (Exact running average)
    if best_match is not None and best_score > SIMILARITY_THRESHOLD:
        stored_embedding_sum = best_match.embedding_sum
        stored_embedding_count = best_match.embedding_count
        updated_embedding_sum = stored_embedding_sum + embedding_sum.cpu()
        updated_embedding_count = stored_embedding_count + len(embeddings)
        updated_average_embedding = updated_embedding_sum / updated_embedding_count
        database.update(
            IdentityClass(
                id=best_match.id,
                name=name,
                role=role,
                embedding=updated_average_embedding.cpu(),
                embedding_sum=updated_embedding_sum.cpu(),
                embedding_count=updated_embedding_count,
                model_name=model_name,
            )
        )
        database.close()
        return f"Successfully updated {name} images"
    else:
        # create a new entry in the database for the new user
        average_embedding = embeddings.mean(dim=0, keepdim=True)
        average_embedding = F.normalize(average_embedding, dim=1)
        database.create(
            IdentityClass(
                name=name,
                role=role,
                embedding=average_embedding.cpu(),
                embedding_sum=embedding_sum.cpu(),
                embedding_count=len(embeddings),
                model_name=model_name,
            )
        )
        database.close()
        return f"Successfully enrolled {name} with {len(embeddings)} images"
