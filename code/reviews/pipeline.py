import json
import os

import pandas as pd

from .apicalls import classify_mentions, get_entities, get_sentiment
from .settings import MENTION_CATEGORIES


def sentiment_step(reviews: pd.DataFrame, cache_file: str) -> list:
    if os.path.exists(cache_file):
        with open(cache_file, "r") as file:
            return json.load(file)

    documents = []

    for review in reviews.to_dict("records"):
        response = get_sentiment(review["text"])
        document = response["results"]["documents"][0]
        document["review_id"] = review["review_id"]
        documents.append(document)

    with open(cache_file, "w") as file:
        json.dump(documents, file)

    return documents


def entities_step(reviews: pd.DataFrame, cache_file: str) -> list:
    if os.path.exists(cache_file):
        with open(cache_file, "r") as file:
            return json.load(file)

    documents = []

    for review in reviews.to_dict("records"):
        response = get_entities(review["text"])
        document = response["results"]["documents"][0]
        document["review_id"] = review["review_id"]
        documents.append(document)

    with open(cache_file, "w") as file:
        json.dump(documents, file)

    return documents


def sentences_table(documents: list) -> pd.DataFrame:
    df = pd.json_normalize(
        documents,
        record_path="sentences",
        meta=["review_id"],
        record_prefix="sentence_"
    )

    df["sentence_id"] = df.groupby("review_id").cumcount()

    df["sentence_text"] = df["sentence_text"].str.strip()

    columns = [
        "review_id",
        "sentence_id",
        "sentence_text",
        "sentence_sentiment",
        "sentence_offset",
        "sentence_length"
    ]

    return df[columns]


def opinions_table(documents: list) -> pd.DataFrame:
    rows = []

    for document in documents:
        for sentence_id, sentence in enumerate(document["sentences"]):
            for target in sentence["targets"]:
                assessments = []

                for relation in target["relations"]:
                    if relation["relationType"] == "assessment":
                        ref = relation["ref"]
                        index = int(ref.split("/")[-1])
                        assessment = sentence["assessments"][index]
                        assessments.append(assessment["text"])

                rows.append({
                    "review_id": document["review_id"],
                    "sentence_id": sentence_id,
                    "mention": target["text"],
                    "mention_sentiment": target["sentiment"],
                    "assessments": ", ".join(assessments),
                    "mention_offset": target["offset"],
                    "source": "opinion"
                })

    columns = [
        "review_id",
        "sentence_id",
        "mention",
        "mention_sentiment",
        "assessments",
        "mention_offset",
        "source"
    ]

    return pd.DataFrame(rows, columns=columns)


def entities_table(documents: list) -> pd.DataFrame:
    df = pd.json_normalize(
        documents,
        record_path="entities",
        meta=["review_id"],
        record_prefix="entity_"
    )

    df = df[df["entity_category"].isin(MENTION_CATEGORIES)]

    columns = [
        "review_id",
        "entity_text",
        "entity_category",
        "entity_offset"
    ]

    df = df[columns]

    return df.reset_index(drop=True)


def subjects_step(reviews: pd.DataFrame, mentions: pd.DataFrame,
                  cache_file: str) -> pd.DataFrame:

    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    rows = []

    for review in reviews.to_dict("records"):
        review_id = review["review_id"]

        review_mentions = mentions[
            mentions["review_id"] == review_id
        ]

        if review_mentions.empty:
            continue

        mention_list = (
            review_mentions
            .sort_values("mention_offset")["mention"]
            .drop_duplicates()
            .tolist()
        )

        response = classify_mentions(review["text"], mention_list)

        for mention in mention_list:
            answer = response["answers"][mention]

            rows.append({
                "review_id": review_id,
                "mention": mention,
                "subject": answer["choice"],
                "subject_confidence": answer["confidence"]
            })

    columns = [
        "review_id",
        "mention",
        "subject",
        "subject_confidence"
    ]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(cache_file, index=False)

    return df
