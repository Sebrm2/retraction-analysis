"""
network_analysis.py
===================
Two network analyses:
  1. Keyword co-occurrence network from paper titles (TF-IDF term selection +
     co-occurrence edges + Louvain community detection).
  2. Co-author retraction network: authors with >= 2 retractions, edges for
     shared retracted papers, communities highlighting collaboration clusters
     that may reflect research groups or paper mills.

Both produce a static PDF/PNG and an interactive HTML (if pyvis is installed).

    python src/network_analysis.py

Depends on data/clean_retractions.parquet.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from config import FIGURES_DIR
from utils import set_style, savefig
from data_prep import load_clean

_STOPWORDS = (
    "the a an and or of in on at to for with by is are was were be been this "
    "that from it its their our has have had study analysis effect role case "
    "report clinical patients results methods using based associated factors "
    "related approach new novel effects between during after before following "
    "human cells type expression disease treatment risk outcomes evidence "
    "randomized controlled trial"
).split()


# ─────────────────────────────────────────────────────────────────────────────
#  KEYWORD CO-OCCURRENCE
# ─────────────────────────────────────────────────────────────────────────────

def keyword_network(df) -> None:
    print("\n== Keyword co-occurrence network ==")
    try:
        import networkx as nx
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("  networkx and scikit-learn required; skipping.")
        return
    set_style()

    titles = df["Title"].dropna().str.lower().tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=_STOPWORDS,
                          max_features=300, min_df=5, token_pattern=r"[a-z][a-z\-]{2,}")
    tfidf = vec.fit_transform(titles)
    vocab = vec.get_feature_names_out()
    scores = np.asarray(tfidf.sum(axis=0)).flatten()
    top_terms = set(vocab[i] for i in scores.argsort()[::-1][:80])

    cooc, freq = Counter(), Counter()
    for title in titles:
        words = list(dict.fromkeys(
            w for w in re.findall(r"[a-z][a-z\-]{2,}", title) if w in top_terms
        ))
        freq.update(words)
        for a, b in combinations(words, 2):
            cooc[tuple(sorted([a, b]))] += 1

    G = nx.Graph()
    for (a, b), w in cooc.items():
        if w >= 3:
            G.add_edge(a, b, weight=w)
    for n in G.nodes():
        G.nodes[n]["freq"] = freq.get(n, 0)
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        print("  Empty graph; skipping.")
        return
    print(f"  Keyword graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    partition = _detect_communities(G)
    _draw_network(G, partition, "fig28_keyword_network",
                  "Keyword co-occurrence network (retracted papers)",
                  size_key="freq", size_scale=4, size_base=80, label_all=True)
    _interactive(G, partition, "fig28_keyword_network_interactive", size_key="freq")


# ─────────────────────────────────────────────────────────────────────────────
#  CO-AUTHOR NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def coauthor_network(df, min_retractions: int = 2) -> None:
    print("\n== Co-author retraction network ==")
    try:
        import networkx as nx
    except ImportError:
        print("  networkx required; skipping.")
        return
    set_style()

    all_authors = df["Author"].dropna().str.split(";").explode().str.strip()
    frequent = set(all_authors.value_counts()[lambda s: s >= min_retractions].index)
    print(f"  Authors with >= {min_retractions} retractions: {len(frequent):,}")

    G = nx.Graph()
    edges = Counter()
    for _, row in df.dropna(subset=["Author"]).iterrows():
        authors = [a.strip() for a in row["Author"].split(";") if a.strip() in frequent]
        for a, b in combinations(authors, 2):
            edges[tuple(sorted([a, b]))] += 1
    for (a, b), w in edges.items():
        G.add_edge(a, b, weight=w)
    counts = all_authors.value_counts()
    for n in list(G.nodes()):
        G.nodes[n]["n_retractions"] = int(counts.get(n, 0))
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        print("  No connected authors; try lowering min_retractions.")
        return
    print(f"  Co-author graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    partition = _detect_communities(G)
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    _draw_network(lcc, partition, "fig29_coauthor_network",
                  "Co-author retraction network (largest component)",
                  size_key="n_retractions", size_scale=60, size_base=40,
                  label_top=30, label_split=True)
    _interactive(lcc, partition, "fig29_coauthor_network_interactive",
                 size_key="n_retractions", label_split=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_communities(G) -> dict:
    try:
        import community as community_louvain
        return community_louvain.best_partition(G)
    except ImportError:
        from networkx.algorithms.community import greedy_modularity_communities
        partition = {}
        for i, comm in enumerate(greedy_modularity_communities(G)):
            for node in comm:
                partition[node] = i
        return partition


def _draw_network(G, partition, name, title, size_key, size_scale, size_base,
                  label_all=False, label_top=None, label_split=False) -> None:
    import networkx as nx
    n_comm = len(set(partition.values()))
    cmap = plt.cm.get_cmap("tab20", max(n_comm, 1))
    fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
    pos = nx.spring_layout(G, k=0.5, seed=42, weight="weight", iterations=50)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3,
                           width=[G[u][v]["weight"] ** 0.6 * 0.6 for u, v in G.edges()],
                           edge_color="#AAAAAA")
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=[G.nodes[n].get(size_key, 1) * size_scale + size_base for n in G.nodes()],
                           node_color=[cmap(partition.get(n, 0)) for n in G.nodes()], alpha=0.85)
    if label_all:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_weight="500", font_color="#222")
    elif label_top:
        top_nodes = sorted(G.degree(), key=lambda x: -x[1])[:label_top]
        labels = {n: (n.split(",")[0] if label_split else n) for n, _ in top_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6.5, font_weight="500")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout(pad=0)
    savefig(name)


def _interactive(G, partition, name, size_key, label_split=False) -> None:
    try:
        from pyvis.network import Network
    except ImportError:
        print("  pyvis not installed; skipping interactive network.")
        return
    colours = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
               "#457B9D", "#6D6875", "#B5838D", "#606C38", "#DDA15E"]
    nt = Network(height="800px", width="100%", bgcolor="#FAFAFA", font_color="#111", notebook=False)
    nt.set_options('{"physics":{"forceAtlas2Based":{"gravitationalConstant":-25,'
                   '"centralGravity":0.002,"springLength":200},"solver":"forceAtlas2Based",'
                   '"minVelocity":0.5},"edges":{"smooth":false}}')
    for n in G.nodes():
        val = G.nodes[n].get(size_key, 1)
        label = n.split(",")[0] if label_split else n
        nt.add_node(n, label=label, color=colours[partition.get(n, 0) % len(colours)],
                    size=max(8, val * 3), title=f"{n} ({val})")
    for u, v, d in G.edges(data=True):
        nt.add_edge(u, v, value=d["weight"])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    nt.save_graph(str(FIGURES_DIR / f"{name}.html"))
    print(f"    -> figures/{name}.html")


def main() -> None:
    df = load_clean()
    keyword_network(df)
    coauthor_network(df, min_retractions=2)
    print("\n  Network analyses complete.")


if __name__ == "__main__":
    main()
