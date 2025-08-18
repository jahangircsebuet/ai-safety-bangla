import json
import random
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class PromptSelectorDiversity:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def load_prompts(self, json_path, min_score=4):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = [item for item in data if item["score"] >= min_score]
        self.prompts = [item["prompt"] for item in self.data]
        print(f"Loaded {len(self.prompts)} high-scoring prompts.")

    def embed_prompts(self):
        self.embeddings = self.model.encode(self.prompts, show_progress_bar=True)

    def cluster_and_sample(self, n_clusters=2000):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(self.embeddings)
        self.selected_prompts = []
        for cluster_id in range(n_clusters):
            idxs = np.where(self.labels == cluster_id)[0]
            if len(idxs) > 0:
                chosen = random.choice(idxs)
                self.selected_prompts.append(self.data[chosen])
        print(f"Selected {len(self.selected_prompts)} diverse prompts.")

    def save_selected(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.selected_prompts, f, ensure_ascii=False, indent=2)

    def visualize_tsne(self, output_png="tsne_visualization.png"):
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(self.embeddings)
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=self.labels, cmap="tab20", s=8)
        plt.title("t-SNE Visualization of Safe Prompt Clusters")
        plt.savefig(output_png, dpi=300)
        print(f"t-SNE plot saved to: {output_png}")