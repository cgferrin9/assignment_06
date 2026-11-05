"""
review_explorer.py — what Salt City Coffee's customers are saying, at a glance.

The owner reads every review on the shop's website, but thirty reviews across
three locations is more than anyone can hold in their head. This page shows
what customers praise and complain about — by location and by subject — and
answers plain-English questions like "Do people like the mocha latte?" or "What
is the most popular product?" from the reviews themselves, with the evidence
underneath.

The page is assembly again: `analyze_reviews` does all the API work (from the
cache after the first run), and the Q&A box is three calls you have already
written — find the question's scope and evidence, build the messages, ask the
model.

Run it:  Run and Debug -> "Streamlit Run: Current File"
Test it: pytest tests/test_pipeline.py -k app
"""

import os

import streamlit as st

from reviews import (
    REVIEWS_FILE,
    analyze_reviews,
    answer_question,
    ask_llm,
    build_messages,
    describe_scope,
    load_aliases,
    summarize_items,
)

# --- The page: the algorithm, step by step ------------------------------------------
#
# Every function this page needs is already in the reviews package (imported above),
# and every widget is one you have used before. Write the code for each step below
# its comment. The tests (pytest tests/test_pipeline.py -k app) check the title, the
# widget keys, the metric labels and values, the tab labels, and the conversation
# the page sends to the model. README Step 8 has the same details as a table.
#
# Run the page as you go (Run and Debug -> "Streamlit Run: Current File") and save
# the file to see each step appear.
#

# 1. WHICH REVIEWS
reviews_file = os.environ.get("REVIEWS_FILE", REVIEWS_FILE)

# 2. HEADING
st.title("Salt City Coffee — What Customers Are Saying")
st.markdown(
    "Every customer review, broken into the things people talk about. "
    "Filter by location or subject, or ask a question at the bottom of the page."
)

# 3. THE DATA
mentions = analyze_reviews(reviews_file)

# 4. TWO FILTERS, SIDE BY SIDE
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        "Location",
        ["All"] + sorted(mentions["location"].dropna().unique().tolist()),
        key="location"
    )

with col2:
    subject = st.selectbox(
        "Subject",
        ["All"] + sorted(mentions["subject"].dropna().unique().tolist()),
        key="subject"
    )

selected = mentions

if location != "All":
    selected = selected[selected["location"] == location]

if subject != "All":
    selected = selected[selected["subject"] == subject]

# 5. FOUR METRICS, SIDE BY SIDE
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Reviews", int(selected["review_id"].nunique()))

with col2:
    st.metric("Mentions", int(len(selected)))

with col3:
    st.metric(
        "Positive",
        int((selected["mention_sentiment"] == "positive").sum())
    )

with col4:
    st.metric(
        "Negative",
        int((selected["mention_sentiment"] == "negative").sum())
    )

# 6. THREE TABLES IN THREE TABS
summary = summarize_items(selected)

tab1, tab2, tab3 = st.tabs([
    "Most praised",
    "Most complained about",
    "What people said"
])

with tab1:
    praised = summary[summary["net"] > 0]
    praised = praised.sort_values("net", ascending=False).head(5)
    st.dataframe(praised, hide_index=True)

with tab2:
    complained = summary[summary["net"] < 0]
    complained = complained.sort_values("net", ascending=True).head(5)
    st.dataframe(complained, hide_index=True)

with tab3:
    st.dataframe(
        selected[
            [
                "date",
                "location",
                "item",
                "subject",
                "mention_sentiment",
                "assessments",
                "sentence_text"
            ]
        ],
        hide_index=True
    )

# 7. THE QUESTION BOX
st.subheader("Ask the reviews")

question = st.text_input(
    "Your question",
    placeholder="What is the most popular product?",
    key="question"
)

# 8. WHEN ASK IS CLICKED
if st.button("Ask", key="ask") and question.strip():

    scope, evidence = answer_question(
        question, mentions, load_aliases()
    )

    if evidence.empty:
        st.warning(
            f"No reviews mention {describe_scope(scope)}. "
            "Try asking about a menu item, staff member, or part of the shop."
        )

    else:
        messages = build_messages(question, evidence)

        with st.spinner("Asking the model..."):
            answer = ask_llm(messages)

        st.markdown(answer)

        st.caption(
            f"Based on {len(evidence)} mentions — "
            f"{describe_scope(scope)}:"
        )

        st.dataframe(
            evidence[
                [
                    "date",
                    "location",
                    "item",
                    "mention_sentiment",
                    "sentence_text"
                ]
            ],
            hide_index=True
        )
