import os
from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.embedding_extract.image_embeddings_extraction import extract_clip_image_embeddings
from src.model.evaluate import evaluate_t5

# Get the absolute path of the current script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # This should point to the root of the project

def get_user_overall_embedding(image_folder_path, prompt_path, alpha, beta):
    """
    Extracts user overall embedding by running image and text embedding extraction in parallel.
    If either image or text folder is missing, only the available embedding is used.
    """
    start_time = datetime.now()
    image_embedding, text_embedding = None, None
    
    # Convert relative paths to absolute paths
    image_folder_path = os.path.join(PROJECT_ROOT, image_folder_path)  # Absolute path to image folder
    prompt_path = os.path.join(PROJECT_ROOT, prompt_path)  # Absolute path to prompt folder

    # Extract image embedding only if the folder exists
    if os.path.exists(image_folder_path):
        def extract_image_embedding():
            print(image_folder_path)
            return extract_clip_image_embeddings(image_folder_path)
    else:
        extract_image_embedding = None  # No image embedding
        print("No image folder found")

    # Extract text embedding only if the text dataset exists
    if os.path.exists(prompt_path):
        print(prompt_path)
        def extract_text_embedding():
            sample_prompt = "I am departing from London, UK in August and will return in March. My budget is $2000. I prefer a historical city destination with tropical weather."
            return evaluate_t5(sample_prompt)
    else:
        extract_text_embedding = None  # No text embedding

    # Run tasks in parallel if both exist
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_image = executor.submit(extract_image_embedding) if extract_image_embedding else None
        future_text = executor.submit(extract_text_embedding) if extract_text_embedding else None

        image_embedding = future_image.result() if future_image else None
        text_embedding = None # future_text.result() if future_text else None
    
    # If only one type of embedding is available, return it directly
    if image_embedding is None and text_embedding is not None:
        return normalize_embedding(text_embedding)
    elif text_embedding is None and image_embedding is not None:
        return normalize_embedding(image_embedding)
    elif image_embedding is None and text_embedding is None:
        raise ValueError("❌ No valid image or text embeddings found!")

    # Normalize embeddings
    image_embedding = normalize_embedding(image_embedding)
    text_embedding = normalize_embedding(text_embedding)
    
    # Ensure both embeddings have the same dimension
    pad_image_embedding, pad_text_embedding = pad_embeddings(image_embedding, text_embedding)
    
    # Compute final user embedding
    final_user_embedding = compute_final_user_embedding(alpha, beta, pad_image_embedding, pad_text_embedding)
    return final_user_embedding


def normalize_embedding(embedding):
    """ Normalizes an embedding to unit length. """
    norm = np.linalg.norm(embedding, keepdims=True)
    return embedding / norm
def pad_embeddings(image_embedding, text_embedding):
    """
    Pads the smaller embedding with zeros to match the larger embedding's dimension.
    """
    img_dim = image_embedding.shape[0]
    text_dim = text_embedding.shape[0]

    dominant_dim = max(img_dim, text_dim)

    padded_image = np.pad(image_embedding, (0, dominant_dim - img_dim), mode='constant') if img_dim < dominant_dim else image_embedding
    padded_text = np.pad(text_embedding, (0, dominant_dim - text_dim), mode='constant') if text_dim < dominant_dim else text_embedding

    return padded_image, padded_text
def compute_final_user_embedding(alpha, beta, pad_image_embedding, pad_text_embedding):
    """ Computes the final weighted user embedding. """
    return alpha * pad_image_embedding + beta * pad_text_embedding


# Example usage
if __name__ == "__main__":
    image_folder_path = "images"
    prompt_path = "synthetic_prompts/tokenized_synthetic_travel_data"
    
    alpha = 0.5
    beta = 0.5
    start_time = datetime.now()
    final_user_embedding = get_user_overall_embedding(image_folder_path, prompt_path, alpha, beta)

    print("Final user embedding shape:", final_user_embedding.shape)
    print("Total time taken:", datetime.now() - start_time)
