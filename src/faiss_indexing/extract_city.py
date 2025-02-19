import os
import faiss
import json
import numpy as np
from src.embedding_extract.implicit_user_embedding import get_user_overall_embedding
import datetime

# Get the absolute path of the current working directory (terminal location)
SCRIPT_DIR = os.getcwd()
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # This should point to the root of the project

def recommend_cities(user_embedding, top_k=None):
    """
    Finds the most similar city embeddings using FAISS and prints similarity scores for all cities.

    Args:
        user_embedding (np.array): The final user embedding vector.
        top_k (int, optional): Number of top cities to retrieve. If None, shows all cities.

    Returns:
        List of recommended city names with similarity scores.
    """
    # Load FAISS index
    index_path = os.path.join(SCRIPT_DIR, "data/embeddings/city_embeddings.index")
    index = faiss.read_index(index_path)

    # Ensure user embedding matches FAISS index dimension
    user_embedding = np.array(user_embedding).astype("float32").reshape(1, -1)
    
    #print(f"User embedding shape: {user_embedding.shape}")
    #print(f"FAISS index dimension: {index.d}")

    if user_embedding.shape[1] < index.d:
        user_embedding = np.pad(user_embedding, ((0, 0), (0, index.d - user_embedding.shape[1])), mode='constant')
    elif user_embedding.shape[1] > index.d:
        user_embedding = user_embedding[:, :index.d]

    #print(f"Adjusted user embedding shape: {user_embedding.shape}")

    # Search FAISS for all city embeddings
    distances, indices = index.search(user_embedding, index.ntotal)  # Retrieve all cities

    # Convert L2 distances to similarity scores (1 / (1 + distance))
    similarity_scores = 1 / (1 + distances[0])

    # Load city names
    city_names_path = os.path.join(SCRIPT_DIR, "data/embeddings/city_names.json")
    with open(city_names_path, "r") as f:
        city_names = json.load(f)

    # Pair city names with similarity scores
    city_scores = [(city_names[idx], similarity_scores[i]) for i, idx in enumerate(indices[0])]

    # Sort by similarity score (descending order)
    city_scores = sorted(city_scores, key=lambda x: x[1], reverse=True)

    # Return top-k cities if specified
    if top_k:
        return city_scores[:top_k]
    return city_scores

# # Example Usage
# start = datetime.datetime.now()
# image_folder_path = os.path.abspath(os.path.join(SCRIPT_DIR, "data/images"))
# #print(image_folder_path)
# prompt_path = os.path.abspath(os.path.join(SCRIPT_DIR, "synthetic_prompts/tokenized_synthetic_travel_data"))
# #print(prompt_path)
# user_embedding = get_user_overall_embedding(image_folder_path, prompt_path, 0.5, 0.5)
# recommendations = recommend_cities(user_embedding, top_k=3)

# print("\n**Top Recommended Cities:**")
# for city, score in recommendations:
#     print(f"{city} - Similarity Score: {score:.4f}")
# end = datetime.datetime.now()
# print("Time taken:", end - start)
