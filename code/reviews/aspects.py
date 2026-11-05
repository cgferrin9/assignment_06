import pandas as pd

from .apicalls import get_entities
from .pipeline import (
    entities_step,
    entities_table,
    opinions_table,
    sentences_table,
    sentiment_step,
    subjects_step,
)
from .settings import (
    ALIASES_FILE,
    LOCATION_WORDS,
    MENTION_CATEGORIES,
    SUBJECT_WORDS,
    SYSTEM_PROMPT,
    cache_path,
)

MENTION_COLUMNS = ["review_id", "sentence_id", "mention", "mention_sentiment",
                   "assessments", "mention_offset", "source"]


def entity_mentions(entities: pd.DataFrame, sentences: pd.DataFrame,
                    opinions: pd.DataFrame) -> pd.DataFrame:
    if entities.empty:
        return pd.DataFrame(columns=MENTION_COLUMNS)
    placed = pd.merge(entities, sentences, on="review_id")
    inside = (placed["entity_offset"] >= placed["sentence_offset"]) & (
        placed["entity_offset"] < placed["sentence_offset"] + placed["sentence_length"]
    )
    placed = placed[inside]

    covered = {}
    for row in opinions.to_dict("records"):
        key = (row["review_id"], row["sentence_id"])
        covered.setdefault(key, []).append(row["mention"].lower())

    def is_covered(row) -> bool:
        """True when an opinion target in the same sentence names this entity."""
        text = row["entity_text"].lower()
        targets = covered.get((row["review_id"], row["sentence_id"]), [])
        return any(text in target or target in text for target in targets)

    keep = placed[~placed.apply(is_covered, axis=1)] if len(placed) else placed
    rows = pd.DataFrame({
        "review_id": keep["review_id"],
        "sentence_id": keep["sentence_id"],
        "mention": keep["entity_text"],
        "mention_sentiment": keep["sentence_sentiment"],
        "assessments": "",
        "mention_offset": keep["entity_offset"],
        "source": "entity",
    }, columns=MENTION_COLUMNS)
    return rows.sort_values(["review_id", "mention_offset"]).reset_index(drop=True)


def load_aliases() -> dict:
    """data/aliases.csv as a dict: alternate spelling -> the item's one name."""
    aliases = pd.read_csv(ALIASES_FILE)
    return dict(zip(aliases["alias"].str.lower(), aliases["item"]))


def item_name(mention: str, aliases: dict) -> str:
    name = mention.lower().strip()
    return aliases.get(name, name)




def build_mentions(reviews: pd.DataFrame, sentences: pd.DataFrame,
                   opinions: pd.DataFrame, entity_rows: pd.DataFrame,
                   subjects: pd.DataFrame, aliases: dict) -> pd.DataFrame:

    combined = pd.concat([opinions, entity_rows], ignore_index=True)

    combined = pd.merge(
        combined, subjects,
        on=["review_id", "mention"],
        how="left"
    )

    combined = pd.merge(
        combined, sentences,
        on=["review_id", "sentence_id"],
        how="left"
    )

    combined = pd.merge(
        combined, reviews,
        on="review_id",
        how="left"
    )

    combined["item"] = combined["mention"].apply(
        lambda x: item_name(x, aliases)
    )

    combined = combined[combined["subject"] != "other"]

    combined = combined.sort_values(
        ["review_id", "mention_offset"]
    )

    combined = combined.drop_duplicates(
        subset=["review_id", "sentence_id", "item"]
    )

    columns = [
        "review_id",
        "date",
        "location",
        "stars",
        "item",
        "mention",
        "subject",
        "mention_sentiment",
        "assessments",
        "sentence_text",
        "source",
        "subject_confidence"
    ]

    return combined[columns].reset_index(drop=True)



def analyze_reviews(reviews_file: str) -> pd.DataFrame:
    reviews = pd.read_csv(reviews_file)

    documents = sentiment_step(
        reviews, cache_path(reviews_file, "sentiment")
    )

    entity_documents = entities_step(
        reviews, cache_path(reviews_file, "entities")
    )

    sentences = sentences_table(documents)
    opinions = opinions_table(documents)
    entities = entities_table(entity_documents)

    entity_rows = entity_mentions(
        entities, sentences, opinions
    )

    combined = pd.concat(
        [opinions, entity_rows], ignore_index=True
    )

    subjects = subjects_step(
        reviews,
        combined,
        cache_path(reviews_file, "subjects", "csv")
    )

    mentions = build_mentions(
        reviews,
        sentences,
        opinions,
        entity_rows,
        subjects,
        load_aliases()
    )

    return mentions



def summarize_items(mentions: pd.DataFrame) -> pd.DataFrame:
    table = pd.pivot_table(
        mentions,
        index=["subject", "item"],
        columns="mention_sentiment",
        aggfunc="size",
        fill_value=0
    )

    table["mentions"] = table.sum(axis=1)

    table["positive"] = table.get("positive", 0)
    table["negative"] = table.get("negative", 0)

    table["net"] = table["positive"] - table["negative"]

    table = table.reset_index()

    columns = [
        "subject",
        "item",
        "mentions",
        "positive",
        "negative",
        "net"
    ]

    table = table[columns]

    table = table.sort_values(
        ["mentions", "item"],
        ascending=[False, True]
    )

    return table.reset_index(drop=True)

# --- The Q&A box --------------------------------------------------------------------


def question_scope(question: str, entity_document: dict, mentions: pd.DataFrame,
                   aliases: dict) -> dict:
    lowered = question.lower()
    candidates = [
        item_name(entity["text"], aliases)
        for entity in entity_document["entities"]
        if entity["category"] in MENTION_CATEGORIES
    ]
    known = sorted(mentions["item"].unique())
    candidates += [item for item in known if item.lower() in lowered]
    items = []
    for item in candidates:
        if item not in items:
            items.append(item)

    def named(word_lists: dict) -> list:
        """The keys with at least one of their words in the question."""
        return [name for name, words in word_lists.items()
                if any(word in lowered for word in words)]

    return {"items": items, "locations": named(LOCATION_WORDS),
            "subjects": named(SUBJECT_WORDS)}


SCOPE_COLUMNS = {"items": "item", "subjects": "subject", "locations": "location"}


def evidence_for(mentions: pd.DataFrame, scope: dict) -> pd.DataFrame:
    selected = mentions
    if scope["items"] or scope["subjects"]:
        what = (mentions["item"].isin(scope["items"])
                | mentions["subject"].isin(scope["subjects"]))
        selected = selected[what]
    if scope["locations"]:
        selected = selected[selected["location"].isin(scope["locations"])]
    return selected.reset_index(drop=True)


def describe_scope(scope: dict) -> str:
    """A scope in words, for the page: "mocha latte · Westcott", or "all reviews"."""
    parts = [", ".join(scope[key]) for key in SCOPE_COLUMNS if scope[key]]
    return " · ".join(parts) if parts else "all reviews"



def build_messages(question: str, evidence: pd.DataFrame, limit: int = 20) -> list:
    total = len(evidence)
    positive = (evidence["mention_sentiment"] == "positive").sum()
    negative = (evidence["mention_sentiment"] == "negative").sum()
    neutral = total - positive - negative

    lines = [
        f"Question: {question}",
        f"Facts: {total} mentions — {positive} positive, {negative} negative, {neutral} neutral or mixed.",
        "Items, most mentioned first:"
    ]

    summary = summarize_items(evidence)

    for row in summary.to_dict("records"):
        lines.append(
            f"- {row['item']} ({row['subject']}): "
            f"{row['mentions']} mentions, "
            f"{row['positive']} positive, "
            f"{row['negative']} negative, "
            f"net {row['net']:+d}"
        )

    lines.append("Review sentences:")

    for row in evidence.head(limit).to_dict("records"):
        lines.append(
            f"- ({row['location']}, {row['date']}, "
            f"{row['item']}, {row['mention_sentiment']}) "
            f"{row['sentence_text']}"
        )

    user_content = "\n".join(lines)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def answer_question(question: str, mentions: pd.DataFrame, aliases: dict):
    """What a question is about and the evidence for it: (scope, evidence)."""
    document = get_entities(question)["results"]["documents"][0]
    scope = question_scope(question, document, mentions, aliases)
    return scope, evidence_for(mentions, scope)
