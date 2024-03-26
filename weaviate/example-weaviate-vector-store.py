#!/bin/env python

import weaviate
from face_recognition import face_encodings, face_locations, load_image_file
import os
import sys


def connect_to_weaviate(host='http://localhost:8080'):
    client = weaviate.Client(url=host)
    return client


def class_exists(client, class_name):
    """Check if a class already exists in the schema."""
    schema = client.schema.get()
    for existing_class in schema.get('classes', []):
        if existing_class.get('class') == class_name:
            return True
    return False


def create_schema(client):
    class_name = "FaceEmbedding"
    if class_exists(client, class_name):
        print(f"Class '{class_name}' already exists.")
        return

    schema = {
        "classes": [
            {
                "class": class_name,
                "vectorizer": "none",  # Explicitly state no vectorizer is used
                "properties": [
                    {
                        "name": "embedding",
                        "dataType": ["vector"],
                    },
                    {
                        "name": "source_info",
                        "dataType": ["text"],
                    },
                ],
            },
        ],
    }
    try:
        client.schema.create(schema)
        print(f"Class '{class_name}' created successfully.")
    except Exception as e:
        print(f"Failed to create class '{class_name}': {str(e)}")

def insert_face_embeddings(client, image_path, model="hog", size_threshold=50):
    known_image = load_image_file(image_path)
    face_locs = face_locations(known_image, model=model)
    
    if not face_locs:
        print("No faces found in the image.")
        return

    large_face_locs = [face_loc for face_loc in face_locs if (face_loc[2] - face_loc[0] > size_threshold) and (face_loc[1] - face_loc[3] > size_threshold)]
    if not large_face_locs:
        print(f"No faces larger than the defined threshold, {size_threshold}, were found.")
        return
    
    face_embeds = face_encodings(known_image, known_face_locations=large_face_locs)
    for embed in face_embeds:
        data_object = {
            "embedding": embed.tolist(),
            "source_info": image_path
        }
        client.data_object.create(data_object, "FaceEmbedding")
    print(f"Inserted {len(face_embeds)} face embeddings from {image_path}.")

def search_face_embeddings(client, embedding, limit=10):
    result = client.query.get("FaceEmbedding", ["embedding", "source_info"]).with_near_vector({"vector": embedding.tolist()}).with_limit(limit).do()
    print("Search results:", result)

if __name__ == "__main__":
    client = connect_to_weaviate()

    try:
        create_schema(client)
        image_path = "../known-faces-02.jpg" if len(sys.argv) == 1 else sys.argv[1]
        insert_face_embeddings(client, image_path, model="hog")
        
        if len(sys.argv) > 1:
            known_image = load_image_file(image_path)
            face_embeddings = face_encodings(known_image)
            if face_embeddings:
                search_face_embeddings(client, face_embeddings[0])
            else:
                print("No embeddings found in the provided image.")
    except Exception as e:
        print(f"An error occurred: {e}")
