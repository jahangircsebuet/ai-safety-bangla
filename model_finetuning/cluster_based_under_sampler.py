from typing import List, Dict, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import logging
import torch
import os

def load_token():
    """
    Load Hugging Face token from environment variable or .env file.
    """
    # Try environment variable first
    token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token
    
    # Try .env files
    env_files = ['.env', '../.env', '../../.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('HUGGING_FACE_HUB_TOKEN='):
                            return line.split('=', 1)[1].strip()
            except Exception:
                continue
    
    # Try dedicated token files
    token_files = ['hf_token.txt', '.hf_token', 'token.txt']
    for token_file in token_files:
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    return f.read().strip()
            except Exception:
                continue
    
    return None

class ClusterCentroidUnderSampler:
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2", seed: int = 42):
        self.seed = seed
        self.model_name = model_name
        
        # Load Hugging Face token
        self.token = load_token()
        if not self.token:
            print("❌ Hugging Face token not found!")
            print("💡 Please set HUGGING_FACE_HUB_TOKEN environment variable or create a .env file")
            raise ValueError("Hugging Face token required for SentenceTransformer")
        
        try:
            # Initialize SentenceTransformer with token
            self.encoder = SentenceTransformer(
                model_name, 
                device="cuda" if torch.cuda.is_available() else "cpu",
                token=self.token
            )
            print(f"✅ SentenceTransformer loaded successfully: {model_name}")
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                print(f"❌ Authentication error: Invalid or missing Hugging Face token.")
                print("💡 Please check your .env file or HUGGING_FACE_HUB_TOKEN environment variable")
            elif "403" in error_msg or "Forbidden" in error_msg:
                print(f"❌ Access denied: You don't have permission to access this model.")
                print("💡 Try using a publicly available model or request access")
            else:
                print(f"❌ Error loading SentenceTransformer: {e}")
            raise
        
        self.logger = logging.getLogger(__name__)

    def balance(self, data: List[Dict], prompt_key: str = "prompt", label_key: str = "label") -> List[Dict]:
        """
        Clusters 'safe' prompts into N clusters, where N = # of 'unsafe'.
        Picks one representative from each cluster:
        closest to centroid by cosine or Euclidean distance.
        Combines with all 'unsafe' items, shuffles.
        """
        df = pd.DataFrame(data)
        unsafe_df = df[df[label_key] == "unsafe"]
        safe_df = df[df[label_key] == "safe"].reset_index(drop=True)
        n = len(unsafe_df)
        if n == 0 or len(safe_df) <= n:
            return data  # no undersampling needed

        prompts = safe_df[prompt_key].tolist()
        embeddings = self.encoder.encode(prompts, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
        self.logger.info(f"Clustering {len(prompts)} safe prompts into {n} groups")
        kmeans = KMeans(n_clusters=n, random_state=self.seed)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        selected_indices = []
        for i in range(n):
            cluster_idxs = np.where(labels == i)[0]
            if len(cluster_idxs) == 0:
                continue
            dists = np.linalg.norm(embeddings[cluster_idxs] - centroids[i], axis=1)
            sel_idx = cluster_idxs[int(np.argmin(dists))]
            selected_indices.append(safe_df.iloc[sel_idx].to_dict())

        safe_selected = pd.DataFrame(selected_indices)
        balanced = pd.concat([unsafe_df, safe_selected], ignore_index=True).sample(frac=1.0, random_state=self.seed)
        return balanced.to_dict(orient="records")
