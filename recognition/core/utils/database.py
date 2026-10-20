import io
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from recognition.config import DATABASE_PATH, DEVICE


class Identities(ABC):
    @abstractmethod
    def get_by_name(self, name):
        raise NotImplementedError

    @abstractmethod
    def get_all(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, identity):
        raise NotImplementedError

    @abstractmethod
    def create(self, identity):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


@dataclass
class IdentityClass:
    name: str
    role: str
    embedding: torch.Tensor
    embedding_sum: torch.Tensor
    embedding_count: int
    model_name: str
    id: Optional[int] = None


class Database(Identities):
    def __init__(self, db_path=DATABASE_PATH):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self._create_table()

    def _create_table(self):
        try:
            self.cursor.execute("""
                           CREATE TABLE IF NOT EXISTS gallery
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               name
                               TEXT
                               NOT
                               NULL,
                               role
                               TEXT
                               NOT
                               NULL,
                               embedding
                               BLOB
                               NOT
                               NULL,
                               embedding_sum
                               BLOB
                               NOT
                               NULL,
                               embedding_count
                               INTEGER
                               NOT
                               NULL
                               DEFAULT
                               0,
                               model_name
                               TEXT
                               NOT
                               NULL
                           )
                           """)
            self.connection.commit()
            print("Database and table verified/created successfully!")
        except sqlite3.Error as error:
            print(f"An error occurred: {error}")

    def create(self, identity: IdentityClass):
        embedding_buffer = io.BytesIO()
        embedding_sum_buffer = io.BytesIO()
        torch.save(identity.embedding, embedding_buffer)
        torch.save(identity.embedding_sum, embedding_sum_buffer)
        blob = embedding_buffer.getvalue()
        embedding_sum = embedding_sum_buffer.getvalue()
        embedding_count = int(identity.embedding_count)



        self.cursor.execute(
            """
            INSERT INTO gallery (name, role, embedding, embedding_sum, embedding_count, model_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                identity.name,
                identity.role,
                blob,
                embedding_sum,
                embedding_count,
                identity.model_name,
            ),
        )
        self.connection.commit()
        print(f"Record created! Assigned ID: {self.cursor.lastrowid}")

    def get_by_name(self, name):
        self.cursor.execute(
            "SELECT id, name, role, embedding, embedding_sum, embedding_count, model_name FROM gallery WHERE name = ?",
            (name,),
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        id_, name, role, embedding, embedding_sum, embedding_count, model_name = row
        embedding = torch.load(io.BytesIO(embedding), weights_only=True)
        embedding_sum = torch.load(io.BytesIO(embedding_sum), weights_only=True)
        embedding_count = int(embedding_count)
        return IdentityClass(
            id=id_,
            name=name,
            role=role,
            embedding=embedding,
            embedding_sum=embedding_sum,
            embedding_count=embedding_count,
            model_name=model_name,
        )

    def update(self, identity: IdentityClass):
        embedding_buffer = io.BytesIO()
        embedding_sum_buffer = io.BytesIO()
        torch.save(identity.embedding, embedding_buffer)
        torch.save(identity.embedding_sum, embedding_sum_buffer)
        blob = embedding_buffer.getvalue()
        embedding_sum = embedding_sum_buffer.getvalue()
        embedding_count = int(identity.embedding_count)
        self.cursor.execute(
            "UPDATE gallery SET embedding = ?, embedding_sum = ?, embedding_count = ?, model_name = ? WHERE id = ?",
            (blob, embedding_sum, embedding_count, identity.model_name, identity.id),
        )
        self.connection.commit()
        print(f"Updated embedding for {identity.name}")

    def get_all(self):
        self.cursor.execute(
            "SELECT id, name, role, embedding, embedding_sum, embedding_count, model_name FROM gallery"
        )
        rows = self.cursor.fetchall()
        results = []
        for row in rows:
            id_, name, role, blob, embedding_sum, embedding_count, model_name = row
            embedding = torch.load(io.BytesIO(blob), weights_only=True)
            embedding_sum = torch.load(io.BytesIO(embedding_sum), weights_only=True)
            embedding_count = int(embedding_count)
            results.append(
                IdentityClass(
                    id=id_,
                    name=name,
                    role=role,
                    embedding=embedding,
                    embedding_sum=embedding_sum,
                    embedding_count=embedding_count,
                    model_name=model_name,
                )
            )
        return results

    def close(self):
        self.connection.close()

    def load_gallery(self):
        return self.get_all()

    @staticmethod
    def check_similarity(gallery: list[IdentityClass], embedding):
        best_match = None
        best_score = -1
        print(f"embedding dimension: {embedding.shape}")

        for user in gallery:
            gallery_embedding = user.embedding.to(DEVICE)
            if gallery_embedding.dim() == 1:
                gallery_embedding = gallery_embedding.unsqueeze(0)
            print(f"gallery-embedding dimension: {gallery_embedding.shape}")
            similarity = torch.mm(embedding, gallery_embedding.T).item()
            print(f"similarity: {similarity}")

            if similarity >= best_score:
                best_score = similarity
                best_match = user

        return best_match, best_score
