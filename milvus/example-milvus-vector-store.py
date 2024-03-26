#!/bin/env python

from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
from face_recognition import face_encodings, face_locations, load_image_file
import os
import sys


def connect_to_milvus(host='localhost', port='19530'):
    connections.connect("default", host=host, port=port)


def create_collection_if_not_exists(collection_name, schema):
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection {collection_name} created.")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection {collection_name} already exists.")
    return collection


def insert_face_embeddings(collection, image_path, model="hog"):
    known_image = load_image_file(image_path)
    face_locs = face_locations(known_image, model=model)
    if not face_locs:
        print("No faces found in the image.")
        return
    
    size_threshold = 50
    large_face_locs = [face_loc for face_loc in face_locs if (face_loc[2] - face_loc[0] > size_threshold) and (face_loc[1] - face_loc[3] > size_threshold)]
    if not large_face_locs:
        print(f"No faces larger than the defined threshold, {size_threshold}, were found.")
        return
    
    face_embeds = face_encodings(known_image, known_face_locations=face_locs)
    for embed in face_embeds:
        # Example: Storing the image filename with the embedding
        collection.insert([[embed.tolist()], [image_path]])
    print(f"Inserted {len(face_embeds)} face embeddings from {image_path}.")


def search_face_embeddings(collection, embedding, search_params={"metric_type": "L2", "params": {"nprobe": 10}}, limit=10):
    results = collection.search([embedding.tolist()], "vector", search_params, limit=limit)
    print("Search results:", results)


if __name__ == "__main__":
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')

    connect_to_milvus(host=MILVUS_HOST, port=MILVUS_PORT)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="source_info", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields, description="Face Embeddings Collection")
    
    collection_name = 'face_embeddings'
    collection = create_collection_if_not_exists(collection_name, schema)
    
    try:
        image_path = "known-faces-01-min.jpg"
        insert_face_embeddings(collection, image_path, model="hog")
        
        # Assuming you want to search using the last inserted embedding
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            known_image = load_image_file(image_path)
            face_embeddings = face_encodings(known_image)
            if face_embeddings:
                search_face_embeddings(collection, face_embeddings[0])
            else:
                print("No embeddings found in the provided image.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        connections.disconnect("default")
