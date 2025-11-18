#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import plotly.express as px
from collections import Counter
from pathlib import Path
from typing import List

import feedparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.cluster import KMeans
import umap


# -------------------------
# Config helpers
# -------------------------
def load_config() -> AzureOpenAI:
    load_dotenv()

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint or not api_key:
        raise RuntimeError(
            "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in environment/.env"
        )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return client


def get_embed_deployment() -> str:
    deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_EMBED_DEPLOYMENT not set in environment/.env")
    return deployment


# -------------------------
# Step 1: parse RSS
# -------------------------
import re
from html import unescape

_HEAR_RE = re.compile(r"האזנה נעימה!?[, ]*רן?", re.UNICODE)


def clean_description(raw: str) -> str:
    if not raw:
        return ""

    # Decode HTML entities (&quot; etc.)
    text = unescape(raw)

    # Remove HTML tags like <a href=...>...</a>, <br>, etc.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs (anything starting with http/https up to whitespace)
    text = re.sub(r"http\S+", " ", text)

    # Fix cases like ".האזנה" -> ". האזנה" (dot stuck to next char)
    text = re.sub(r"\.(?=[^\s])", ". ", text)

    # Remove 'האזנה נעימה! רן' (with optional ! , and spaces)
    text = _HEAR_RE.sub(" ", text)

    # Collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_rss(rss_path: Path) -> pd.DataFrame:
    print(f"[parse] Reading RSS from {rss_path}")
    feed = feedparser.parse(rss_path.as_posix())

    rows = []
    for idx, entry in enumerate(feed.entries):
        title = (entry.get("title", "").strip())
        # Remove leading prefix like [עושים היסטוריה]
        tmp = re.sub(r"^\s*\[.*?]\s*", "", title)

        # Remove leading number + colon e.g. 276: or 455:
        tmp = re.sub(r"^\s*\d+\s*:\s*", "", tmp)

        # Remove trailing bracket section if exists (rare cases)
        title = re.sub(r"\s*\[.*?]\s*$", "", tmp).strip()

        title = (title.strip()
                 .replace("(ש.ח.)", "")
                 .replace('[עושים היסטוריה]', '')
                 )

        raw_desc = (
                getattr(entry, "itunes_summary", None)
                or entry.get("summary", "")
                or ""
        )

        desc = clean_description(raw_desc)

        guid = entry.get("id") or entry.get("guid") or f"item-{idx}"
        link = entry.get("link", "")

        text = f"{title}. {desc}".strip()

        rows.append(
            {
                "guid": guid,
                "title": title,
                "description": desc,
                "link": link,
                "text": text,
            }
        )

    df = pd.DataFrame(rows)
    print(f"[parse] Parsed {len(df)} episodes")
    return df


# -------------------------
# Step 2: embeddings
# -------------------------
def embed_texts(
        client: AzureOpenAI,
        deployment: str,
        texts: List[str],
        batch_size: int = 64,
) -> np.ndarray:
    print(f"[embed] Embedding {len(texts)} texts with deployment '{deployment}'")
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        print(f"[embed] Batch {i} - {i + len(batch)}")
        resp = client.embeddings.create(
            model=deployment,
            input=batch,
        )
        for item in resp.data:
            all_embeddings.append(item.embedding)

    X = np.array(all_embeddings, dtype="float32")
    print(f"[embed] Shape: {X.shape}")
    return X


# -------------------------
# Step 3: clustering
# -------------------------
def cluster_embeddings(
        X: np.ndarray,
        n_clusters: int = 20,
        random_state: int = 42,
) -> np.ndarray:
    print(f"[cluster] KMeans with n_clusters={n_clusters}")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return labels


# -------------------------
# Step 4: 2D projection & plot (UMAP)
# -------------------------
def project_to_2d(X: np.ndarray) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D using UMAP
    with cosine distance – better than PCA for semantic embeddings.
    """
    print("[umap] Projecting to 2D with UMAP (cosine)")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    X2 = reducer.fit_transform(X)
    print(f"[umap] Resulting shape: {X2.shape}")
    return X2


def project_to_3d(X: np.ndarray) -> np.ndarray:
    """
    Project high-dimensional embeddings to 3D using UMAP
    with cosine distance – suitable for 3D interactive plots.
    """
    print("[umap] Projecting to 3D with UMAP (cosine)")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    X3 = reducer.fit_transform(X)
    print(f"[umap] Resulting shape: {X3.shape}")
    return X3


def plot_clusters(
        X2: np.ndarray,
        labels: np.ndarray,
        titles: List[str],
        out_path: Path,
):
    print(f"[plot] Saving cluster scatter to {out_path}")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=20, alpha=0.8)
    plt.title("Osim Historia – Episode Clusters (UMAP + KMeans)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Legend with cluster IDs
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.8)
    legend_labels = [f"Cluster {i}" for i in sorted(set(labels))]
    plt.legend(handles, legend_labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_clusters_interactive(
        X2: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame,
        out_path: Path,
) -> None:
    """
    Create an interactive 2D map (HTML) of the clusters using Plotly.

    - Color = cluster id
    - Hover = title + description (+ link)
    """
    print(f"[plot] Saving interactive cluster map to {out_path}")

    plot_df = pd.DataFrame(
        {
            "x": X2[:, 0],
            "y": X2[:, 1],
            "cluster": labels,
            "title": df["title"].values,
            "description": df["description"].values,
            "link": df["link"].values,
        }
    )

    # Build interactive scatter
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster",
        hover_data={
            "cluster": True,
            "title": True,
            "description": False,
            "link": False,
            "x": False,
            "y": False,
        },
        title="Osim Historia – Episode Map (UMAP + KMeans)",
    )

    # Make it look polished
    fig.update_layout(
        template="plotly_white",
        width=1000,
        height=800,
        legend_title_text="Cluster",
        margin=dict(l=40, r=40, t=60, b=40),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
    )

    fig.update_traces(
        marker=dict(
            size=8,
            line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
            opacity=0.85,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Standalone HTML file with all JS inside
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


def plot_clusters_interactive_3d(
    X3: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Create an interactive 3D map (HTML) of the clusters using Plotly.

    - Axes = UMAP-1/2/3
    - Color = cluster id
    - Hover = title + description (+ link)
    """
    print(f"[plot] Saving interactive 3D cluster map to {out_path}")

    plot_df = pd.DataFrame(
        {
            "x": X3[:, 0],
            "y": X3[:, 1],
            "z": X3[:, 2],
            "cluster": labels,
            "title": df["title"].values,
            "description": df["description"].values,
            "link": df["link"].values,
        }
    )

    fig = px.scatter_3d(
        plot_df,
        x="x",
        y="y",
        z="z",
        color="cluster",
        hover_data={
            "cluster": True,
            "title": True,
            "description": False,
            "link": False,
            "x": False,
            "y": False,
            "z": False,
        },
        title="Osim Historia – Episode Map (3D UMAP + KMeans)",
    )

    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=800,
        legend_title_text="Cluster",
        margin=dict(l=40, r=40, t=60, b=40),
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
    )

    fig.update_traces(
        marker=dict(
            size=6,
            line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
            opacity=0.75,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


# -------------------------
# Step 5: simple analysis
# -------------------------
_STOPWORDS = {
    # English
    "the", "a", "an", "in", "on", "of", "and", "for", "to", "with", "is", "are",
    "from", "at", "by", "about", "this", "that", "it", "as", "be", "or",
    # Hebrew-ish basics – not perfect but helps
    "על", "של", "את", "עם", "לא", "כן", "גם", "הוא", "היא", "הם", "הן", "אנחנו",
    "אם", "אבל", "עוד", "מאוד", "בלי", "כל", "אז", "מן",
}


def tokenize(text: str) -> List[str]:
    # very simple tokenization on non-letters
    import re

    tokens = re.split(r"[^\wא-ת]+", text.lower())
    return [t for t in tokens if t and t not in _STOPWORDS]


def analyze_clusters(
        df: pd.DataFrame,
        labels: np.ndarray,
        out_path: Path,
        top_n_titles: int = 5,
        top_n_words: int = 8,
):
    print(f"[analyze] Writing summary to {out_path}")
    df = df.copy()
    df["cluster"] = labels

    lines: List[str] = []
    n_clusters = len(sorted(set(labels)))
    lines.append(f"Total episodes: {len(df)}")
    lines.append(f"Total clusters: {n_clusters}")
    lines.append("")

    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        lines.append("=" * 80)
        lines.append(f"Cluster {cluster_id} – {len(sub)} episodes")
        lines.append("-" * 80)

        # Top titles (just first N)
        lines.append("Sample titles:")
        for _, row in sub.head(top_n_titles).iterrows():
            lines.append(f"  • {row['title']}")

        # Keyword summary
        all_text = " ".join(sub["text"].astype(str).tolist())
        words = tokenize(all_text)
        counts = Counter(words)
        common = counts.most_common(top_n_words)
        if common:
            lines.append("Top keywords:")
            lines.append(
                "  "
                + ", ".join(f"{w} ({c})" for w, c in common)
            )
        else:
            lines.append("Top keywords: (none)")

        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8-sig")

    # Also print a short version to console
    print("\n[analyze] Short overview:")
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        print(f"  Cluster {cluster_id}: {len(sub)} episodes")


# -------------------------
# Orchestration
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cluster ‘Osim Historia’ episodes from an RSS feed using Azure embeddings."
    )
    parser.add_argument(
        "--rss",
        type=str,
        default="data/feed.rss",
        help="Path to RSS file (default: data/feed.rss)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=8,
        help="Number of KMeans clusters (default: 5)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load episodes and embeddings from out/episodes.csv and out/embeddings.npy if they exist.",
    )

    args = parser.parse_args()

    rss_path = Path(args.rss)
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    episodes_csv = out_dir / "episodes.csv"
    embeddings_npy = out_dir / "embeddings.npy"

    # 1) Load from cache if requested and available
    if args.use_cache and episodes_csv.exists() and embeddings_npy.exists():
        print(f"[cache] Loading episodes from {episodes_csv}")
        df = pd.read_csv(episodes_csv)

        print(f"[cache] Loading embeddings from {embeddings_npy}")
        X = np.load(embeddings_npy)

        if len(df) != X.shape[0]:
            print(
                f"[cache] Mismatch between episodes ({len(df)}) and embeddings ({X.shape[0]}). "
                "Ignoring cache and recomputing."
            )
            df = None
            X = None
    else:
        df = None
        X = None

    # 2) If cache is not used or invalid, recompute
    if df is None or X is None:
        if not rss_path.exists():
            raise FileNotFoundError(f"RSS file not found: {rss_path}")

        print(f"[parse] Reading RSS from {rss_path}")
        df = parse_rss(rss_path)
        csv_path = episodes_csv
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[parse] Saved episodes to {csv_path}")

        client = load_config()
        deployment = get_embed_deployment()
        X = embed_texts(client, deployment, df["text"].astype(str).tolist())
        np.save(embeddings_npy, X)
        print(f"[embed] Saved embeddings to {embeddings_npy}")

    # 3) Cluster
    labels = cluster_embeddings(X, n_clusters=args.clusters)
    df["cluster"] = labels
    clusters_csv = out_dir / "episodes_with_clusters.csv"
    df.to_csv(clusters_csv, index=False, encoding="utf-8-sig")
    print(f"[cluster] Saved clustered episodes to {clusters_csv}")

    # 4) UMAP 2D + 3D projections and interactive plots
    # 2D map
    X2 = project_to_2d(X)
    html_2d = out_dir / "clusters_umap_2d.html"
    plot_clusters_interactive(X2, labels, df, html_2d)

    # 3D map
    X3 = project_to_3d(X)
    html_3d = out_dir / "clusters_umap_3d.html"
    plot_clusters_interactive_3d(X3, labels, df, html_3d)

    # Interactive HTML
    html_path = out_dir / "clusters_umap.html"
    plot_clusters_interactive(X2, labels, df, html_path)

    # 5) Simple text-based analysis
    summary_path = out_dir / "cluster_summary.txt"
    analyze_clusters(df, labels, summary_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
