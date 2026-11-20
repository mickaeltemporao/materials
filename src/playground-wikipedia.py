"""
==============================================================================
Playground: Wikipedia Revision History & Sentiment Analysis
==============================================================================

Fetch recent revisions of a Wikipedia page, extract line diffs between
consecutive edits, score the sentiment of edits with VADER, and visualize
sentiment over time with Altair.

Execute line by line using Shift+Enter in VS Code.
==============================================================================
"""

import datetime
import difflib
import mwclient
import nltk
import pandas as pd
import altair as alt
import statsmodels.formula.api as smf
from nltk.sentiment import SentimentIntensityAnalyzer
from statsmodels.nonparametric.smoothers_lowess import lowess

nltk.download("vader_lexicon", quiet=True)


# ==============================================================================
# Step 1: Helper Functions for Wikipedia Diffs & Sentiment
# ==============================================================================
def fetch_wikipedia_revisions(page_title, limit=20):
    site = mwclient.Site("en.wikipedia.org")
    page = site.pages[page_title]
    revs = page.revisions(
        prop="content|timestamp",
        max_items=limit,
        api_chunk_size=limit,
    )
    return list(revs)


def compare_revisions(rev_old, rev_new):
    diff = difflib.unified_diff(
        rev_old.splitlines(),
        rev_new.splitlines(),
        lineterm="",
    )
    return "\n".join(diff)


def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)


# ==============================================================================
# Step 2: Fetch Revisions & Compute Edit Sentiment
# ==============================================================================
page_title = "Gabriel Attal"
revisions = fetch_wikipedia_revisions(page_title, limit=20)

rows = []
for i in range(len(revisions) - 1):
    rev_new = revisions[i].get("*", "")
    rev_old = revisions[i + 1].get("*", "")
    changes = compare_revisions(rev_old, rev_new)
    scores = sentiment_analysis(changes)
    ts = revisions[i]["timestamp"]
    rows.append({
        "date": datetime.datetime(*ts[:6]),
        "compound": scores["compound"],
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
df["step"] = df.index
df["rolling_avg"] = df["compound"].rolling(window=5, min_periods=1).mean()
df["smoothed"] = lowess(
    df["compound"], df["step"], frac=0.6, it=0, return_sorted=False
)

df.head()


# ==============================================================================
# Step 3: Visualize Sentiment Evolution with Altair
# ==============================================================================
rolling_line = (
    alt.Chart(df)
    .mark_line(color="#4c78a8", point=True)
    .encode(
        x=alt.X("date", type="temporal", title="Revision Date"),
        y=alt.Y(
            "rolling_avg",
            type="quantitative",
            title="Sentiment Score (VADER Compound)",
        ),
    )
)

smooth_line = (
    alt.Chart(df)
    .mark_line(color="#e45756", strokeWidth=3)
    .encode(
        x=alt.X("date", type="temporal"),
        y=alt.Y("smoothed", type="quantitative"),
    )
)

sentiment_chart = (
    (rolling_line + smooth_line)
    .properties(
        title=f"Sentiment of Wikipedia Revisions: {page_title}",
        width=500,
        height=300,
    )
)
sentiment_chart


# ==============================================================================
# Step 4: Simple Trend Model with statsmodels Formula API
# ==============================================================================
trend_model = smf.ols(formula="compound ~ step", data=df).fit()
trend_model.summary()
