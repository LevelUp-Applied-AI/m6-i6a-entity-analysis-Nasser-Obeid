"""
Module 6 Week A — Integration: Entity Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from itertools import combinations


def load_and_filter_corpus(filepath="data/climate_articles.csv"):
    df = pd.read_csv(filepath)
    return df[df["language"] == "en"].reset_index(drop=True)


def run_ner_pipeline(texts):
    nlp = spacy.load("en_core_web_sm")
    rows = []
    for text_id, text_string in texts:
        doc = nlp(text_string)
        for ent in doc.ents:
            rows.append({"text_id": text_id, "entity_text": ent.text, "entity_label": ent.label_})
    return pd.DataFrame(rows, columns=["text_id", "entity_text", "entity_label"])


def aggregate_entity_stats(entity_df):
    freq = (
        entity_df.groupby(["entity_text", "entity_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    label_counts = entity_df.groupby("entity_label").size().to_dict()
    co_rows = []
    for text_id, group in entity_df.groupby("text_id"):
        unique_entities = group["entity_text"].unique().tolist()
        for a, b in combinations(sorted(unique_entities), 2):
            co_rows.append({"entity_a": a, "entity_b": b})
    if co_rows:
        co_df = (
            pd.DataFrame(co_rows)
            .groupby(["entity_a", "entity_b"])
            .size()
            .reset_index(name="co_count")
            .sort_values("co_count", ascending=False)
            .reset_index(drop=True)
        )
    else:
        co_df = pd.DataFrame(columns=["entity_a", "entity_b", "co_count"])
    return {"top_entities": freq, "label_counts": label_counts, "co_occurrence": co_df}


def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    top = stats["top_entities"].copy()
    labels = top["entity_label"].unique()
    color_map = {label: plt.cm.tab10(i) for i, label in enumerate(labels)}
    colors = top["entity_label"].map(color_map)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["entity_text"], top["count"], color=colors)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Entity")
    ax.set_title("Top 20 Named Entities by Frequency")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[label]) for label in labels]
    ax.legend(handles, labels, title="Entity Type", loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_report(stats, co_occurrence):
    lines = ["=== Entity Analysis Report ===\n", "Entity Counts by Type:"]
    for label, count in sorted(stats["label_counts"].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {label}: {count}")
    lines.append("\nTop 5 Most Frequent Entities:")
    for _, row in stats["top_entities"].head(5).iterrows():
        lines.append(f"  {row['entity_text']} ({row['entity_label']}): {row['count']}")
    lines.append("\nTop 3 Co-occurring Entity Pairs:")
    if co_occurrence is not None and len(co_occurrence) > 0:
        for _, row in co_occurrence.head(3).iterrows():
            lines.append(f"  '{row['entity_a']}' & '{row['entity_b']}': {row['co_count']} texts")
    else:
        lines.append("  No co-occurrence data available.")
    total_entities = sum(stats["label_counts"].values())
    top_entity = stats["top_entities"].iloc[0]
    lines.append(f"\nSummary: Extracted {total_entities} total entity mentions. Most frequent: '{top_entity['entity_text']}' ({top_entity['entity_label']}) with {top_entity['count']} occurrences.")
    return "\n".join(lines)


if __name__ == "__main__":
    corpus = load_and_filter_corpus()
    if corpus is not None:
        print(f"Corpus: {len(corpus)} English articles")
        print(f"Categories: {corpus['category'].value_counts().to_dict()}")
        texts = list(zip(corpus["id"], corpus["text"]))
        entities = run_ner_pipeline(texts)
        if entities is not None:
            print(f"\nExtracted {len(entities)} entities")
            stats = aggregate_entity_stats(entities)
            if stats is not None:
                print(f"\nLabel counts: {stats['label_counts']}")
                print(f"\nTop 5 entities:")
                print(stats["top_entities"].head())
                visualize_entity_distribution(stats)
                print("\nVisualization saved to entity_distribution.png")
                report = generate_report(stats, stats.get("co_occurrence"))
                if report is not None:
                    print(f"\n{'='*50}")
                    print(report)