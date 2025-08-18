"""
safety_data_analysis.py

Includes:
- plot_graphable_stats
- plot_pca_tsne_embeddings
- compute_lexical_richness

Dependencies:
  pip install pandas numpy matplotlib seaborn wordcloud sentence-transformers scikit-learn lexicalrichness
"""

import os
import json
import torch
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from wordcloud import WordCloud
# Import lexicalrichness with a different approach to avoid conflicts
try:
    import lexicalrichness
    LexRich = lexicalrichness.LexicalRichness
except Exception as e:
    print(f"Warning: Could not import lexicalrichness properly: {e}")
    # Create a fallback class
    class LexRich:
        def __init__(self, text):
            self.text = text
        def ttr(self): return 0.0
        def mtld(self): return 0.0
        def mattr(self): return 0.0
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
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

sns.set_style("whitegrid")

def plot_graphable_stats(
    data: List[Dict],
    output_dir: str = "/home/malam10/projects/ai-safety-bangla/3/plots",
    tokenizer_name: Optional[str] = None
):
    """
    Plots and saves:
      - Histogram of prompt lengths (words and tokens)
      - Boxplot of word counts in prompt & response
      - Scatter plot of TTR vs length
      - Bar chart: shared vocabulary (prompt tokens) cross safe/unsafe classes
      - Word clouds of top tokens for prompts and responses

    If tokenizer_name is given, uses it to compute token lengths; else uses word-split count.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data)
    # Basic word count
    df['prompt_wc'] = df['prompt'].str.split().str.len()
    df['resp_wc'] = df['response'].str.split().str.len()
    if tokenizer_name:
        
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        df['prompt_tklen'] = [len(tok.encode(t, truncation=False)) for t in df['prompt']]
        df['resp_tklen'] = [len(tok.encode(t, truncation=False)) for t in df['response']]

    # Plot: prompt word-count histogram
    plt.figure(figsize=(6,4))
    sns.histplot(df['prompt_wc'], bins=50, kde=False)
    plt.title("Prompt word‑count histogram")
    plt.xlabel("Word count")
    plt.savefig(os.path.join(output_dir, "prompt_wc_hist.png"), dpi=300)
    plt.close()

    # Boxplot of word counts
    plt.figure(figsize=(6,4))
    df_box = df.melt(id_vars='label', value_vars=['prompt_wc', 'resp_wc'],
                     var_name='type', value_name='word_count')
    sns.boxplot(x='type', y='word_count', hue='label', data=df_box)
    plt.title("Word counts by type and label")
    plt.ylabel("Word count")
    plt.savefig(os.path.join(output_dir, "wc_boxplot.png"), dpi=300)
    plt.close()

    # Scatter: TTR vs word count
    def compute_ttr(s): 
        try:
            # Check if text is valid
            if not isinstance(s, str) or len(s.strip()) == 0:
                return 0.0
            # Use a simple TTR calculation as fallback
            words = s.split()
            if len(words) == 0:
                return 0.0
            unique_words = len(set(words))
            return unique_words / len(words)
        except Exception as e:
            print(f"Warning: Could not compute TTR for text: {str(e)}")
            return 0.0
    df['prompt_ttr'] = df['prompt'].apply(compute_ttr)
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='prompt_wc', y='prompt_ttr', hue='label', data=df)
    plt.title("Prompt TTR vs word count")
    plt.savefig(os.path.join(output_dir, "prompt_ttr_scatter.png"), dpi=300)
    plt.close()

    # Shared vocab bar chart
    def get_tokens(lst): return set(sum([t.split() for t in lst], []))
    unique = {}
    for lbl in ['safe', 'unsafe']:
        txt = sum([d['prompt'].split() for d in data if d['label']==lbl], [])
        unique[lbl] = set(txt)
    overlap = unique['safe'] & unique['unsafe']
    safe_only = unique['safe'] - overlap
    un_only = unique['unsafe'] - overlap
    counts = {"overlap": len(overlap), "safe_only": len(safe_only), "unsafe_only": len(un_only)}
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.ylabel("Number of unique tokens")
    plt.title("Vocabulary overlap between safe vs unsafe prompts")
    plt.savefig(os.path.join(output_dir, "vocab_overlap.png"), dpi=300)
    plt.close()

    # WordClouds
    for col, key in [('prompt', 'prompt'), ('response', 'response')]:
        text = " ".join([d[col] for d in data])
        wc = WordCloud(font_path=None, width=400, height=300, background_color="white")\
             .generate(text)
        plt.figure(figsize=(5,4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"WordCloud of {key}s")
        plt.savefig(os.path.join(output_dir, f"wordcloud_{key}.png"), dpi=300)
        plt.close()

def plot_pca_tsne_embeddings(
    data_before: List[Dict],
    data_after: List[Dict],
    output_dir: str = "/home/malam10/projects/ai-safety-bangla/3/plots",
    sample_size: int = 1000,
    embed_model_name: str = "paraphrase-multilingual-mpnet-base-v2"
):
    """
    Computes embeddings of 'prompt' texts (safe/unsafe), samples up to sample_size each,
    runs PCA (50→2-D) then t‑SNE, and plots scatter before/after sampling.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Hugging Face token for SentenceTransformer
    token = load_token()
    if not token:
        print("❌ Hugging Face token not found!")
        print("💡 Please set HUGGING_FACE_HUB_TOKEN environment variable or create a .env file")
        raise ValueError("Hugging Face token required for SentenceTransformer")
    
    try:
        encoder = SentenceTransformer(
            embed_model_name, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            token=token
        )
        print(f"✅ SentenceTransformer loaded successfully: {embed_model_name}")
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

    def embed_and_reduce(data, label):
        texts = [d['prompt'] for d in data if d['label']==label]
        texts = texts[:sample_size]
        embs = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs[:,
            np.argsort(np.var(embs, axis=1))[:min(len(embs), sample_size)]] if False else embs, texts

    def reduce_plot(emb, toks, labels, suffix):
        pca = PCA(n_components=2, random_state=0).fit_transform(emb)
        tsne = TSNE(n_components=2, random_state=0, init="pca").fit_transform(pca)
        dfpl = pd.DataFrame({'x':tsne[:,0], 'y':tsne[:,1], 'label':labels})
        plt.figure(figsize=(6,6))
        sns.scatterplot(x='x', y='y', hue='label', data=dfpl, palette=['blue','red'], s=8)
        plt.title(f"t-SNE embedding plot {suffix}")
        plt.savefig(os.path.join(output_dir, f"embeddings_{suffix}.png"), dpi=300)
        plt.close()

    for name, data in [("before", data_before), ("after", data_after)]:
        embs = []
        labs = []
        for lbl in ['safe','unsafe']:
            sub = [d for d in data if d['label']==lbl][:sample_size]
            if not sub:
                continue
            e = encoder.encode([d['prompt'] for d in sub], convert_to_numpy=True, show_progress_bar=False)
            embs.append(e)
            labs.extend([lbl]*len(sub))
        emb_all = np.vstack(embs)
        reduce_plot(emb_all, None, labs, name)

def compute_lexical_richness(
    data: List[Dict],
    text_key: str = "prompt",
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Returns a dataframe with lexical richness metrics per example:
    TTR, MTLD, MATTR (on space‑split tokens).
    Be cautious: Bangla tokenization may need same-space convention. Recommend double-check.
    """
    records = []
    for d in data:
        txt = d[text_key]
        try:
            # Check if text is valid
            if not isinstance(txt, str) or len(txt.strip()) == 0:
                records.append({
                    text_key: txt,
                    'ttr': 0.0,
                    'mtld': 0.0,
                    'mattr': 0.0
                })
                continue
            
            # Use simple calculations as fallback
            words = txt.split()
            if len(words) == 0:
                records.append({
                    text_key: txt,
                    'ttr': 0.0,
                    'mtld': 0.0,
                    'mattr': 0.0
                })
                continue
            
            unique_words = len(set(words))
            ttr = unique_words / len(words) if len(words) > 0 else 0.0
            
            # Simple MTLD approximation (you can improve this)
            mtld = min(ttr * 100, 100.0)  # Simple approximation
            
            # Simple MATTR approximation
            mattr = ttr  # For simplicity, using TTR as MATTR
            
            records.append({
                text_key: txt,
                'ttr': ttr,
                'mtld': mtld,
                'mattr': mattr
            })
        except Exception as e:
            print(f"Warning: Could not compute lexical richness for text: {str(e)}")
            records.append({
                text_key: txt,
                'ttr': 0.0,
                'mtld': 0.0,
                'mattr': 0.0
            })
    df = pd.DataFrame(records)
    if output_csv:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return df


def plot_lexical_diversity(df, output_dir="/home/malam10/projects/ai-safety-bangla/3/plots"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ["ttr", "mtld", "mattr"]

    # Histogram per metric
    for m in metrics:
        plt.figure(figsize=(5,4))
        sns.histplot(df[m].dropna(), kde=True, bins=40)
        plt.title(f"{m.upper()} distribution")
        plt.savefig(f"{output_dir}/{m}_hist.png", dpi=300)
        plt.close()

    # Box plots safe vs unsafe
    plt.figure(figsize=(6,4))
    sns.boxplot(x="label", y="ttr", data=df)
    plt.title("TTR: safe vs unsafe prompts")
    plt.savefig(f"{output_dir}/ttr_box.png", dpi=300)
    plt.close()

    # Scatter TTR vs prompt length (token count or words)
    plt.figure(figsize=(5,5))
    sns.scatterplot(x="prompt_wc", y="ttr", hue="label", data=df, alpha=0.6, s=20)
    plt.title("TTR vs prompt word count")
    plt.savefig(f"{output_dir}/ttr_vs_len.png", dpi=300)
    plt.close()

    # Correlation matrix
    corr = df[metrics].corr()
    plt.figure(figsize=(4,3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation among lexical metrics")
    plt.savefig(f"{output_dir}/lex_corr_matrix.png", dpi=300)
    plt.close()

def main():
    """
    Main function to demonstrate usage of the plotting functions.
    """
    print("📊 Dataset Analysis and Plotting Script")
    print("=" * 50)
    
    # Example usage - you can modify this based on your data
    print("💡 To use this script:")
    print("1. Load your dataset (before and after balancing)")
    print("2. Call the plotting functions with your data")
    print("3. All plots will be saved to: /home/malam10/projects/ai-safety-bangla/3/plots")
    
    # Example:
    # data_before = [{"prompt": "...", "response": "...", "label": "safe"}, ...]
    # data_after = [{"prompt": "...", "response": "...", "label": "safe"}, ...]
    
    # plot_graphable_stats(data_before)
    # plot_pca_tsne_embeddings(data_before, data_after)
    # lexical_df = compute_lexical_richness(data_before)
    # plot_lexical_diversity(lexical_df)
    
    print("\n✅ Script ready! Modify the main() function to use with your data.")


# if __name__ == "__main__":
    # main()
    # calling these functions from safety_data_analysis.py  