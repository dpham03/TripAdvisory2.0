import os
import clip
import torch
import pandas as pd
from PIL import Image
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Set the image folder
image_folder = "./images"

# Initialize a list to store results
results = []

# Process images and store embeddings
image_embeddings = []

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)

    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Compute the image embedding
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)  # Normalize embedding

        # Convert to numpy and store
        embedding_np = image_feature.cpu().numpy().flatten()
        image_embeddings.append(embedding_np)

        # Store results
        results.append([filename, embedding_np])

        print(f"Processed: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Convert results to DataFrame
df = pd.DataFrame(results, columns=["Image Name", "Embedding"])

# Aggregate embeddings (if needed)
if len(image_embeddings) > 1:
    aggregated_embedding = np.mean(image_embeddings, axis=0)  # Average pooling
else:
    aggregated_embedding = image_embeddings[0] if image_embeddings else None

# Save embeddings to CSV (for easy lookup)
df.to_csv("image_embeddings.csv", index=False)


# Print aggregated embedding shape
if aggregated_embedding is not None:
    print("\nAggregated Embedding Shape:", aggregated_embedding.shape)  # Should be (512,)
