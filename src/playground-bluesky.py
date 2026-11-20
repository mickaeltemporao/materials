"""
==============================================================================
Playground: Bluesky Data Collection, Text Analysis & Modeling
==============================================================================

Execute line by line using Shift+Enter in VS Code.
==============================================================================
"""

import requests
import pandas as pd
import altair as alt
import statsmodels.formula.api as smf


# ==============================================================================
# Step 1: Fetch Posts from Bluesky Public API (No API Key Needed)
# ==============================================================================
# Note: Emmanuel Macron has no official personal Bluesky account.
# Swap handle for any public account (e.g. "lemonde.fr", "afp.com"):
handle = "info.gouv.fr"
url = (
    "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
    f"?actor={handle}&limit=50"
)

data = requests.get(url).json()


# ==============================================================================
# Step 2: Extract Text, Likes, and Shares
# ==============================================================================
posts = []
for item in data.get("feed", []):
    post = item["post"]
    posts.append({
        "text": post["record"].get("text", ""),
        "likes": post.get("likeCount", 0),
        "shares": post.get("repostCount", 0),
    })


# ==============================================================================
# Step 3: Save to File & Load into a DataFrame
# ==============================================================================
pd.DataFrame(posts).to_csv("bluesky_posts.csv", index=False)

df = pd.read_csv("bluesky_posts.csv")
df.head()


# ==============================================================================
# Step 4: Count Keyword Occurrences & Filter
# ==============================================================================
# Check whether each post mentions "étudiant" (case-insensitive):
df["mentions_word"] = df["text"].str.contains(
    "étudiant", case=False, na=False
)
df["mentions_word"].sum()

# View matching posts:
df[df["mentions_word"]]


# ==============================================================================
# Step 5: Visualize Engagement with Altair & Simple Model
# ==============================================================================
engagement_chart = (
    alt.Chart(df)
    .mark_circle(size=60, opacity=0.7)
    .encode(
        x=alt.X("shares", type="quantitative", title="Shares (Reposts)"),
        y=alt.Y("likes", type="quantitative", title="Likes"),
        color=alt.Color(
            "mentions_word",
            type="nominal",
            title="Mentions Keyword",
        ),
    )
    .properties(
        title="Bluesky Post Engagement: Likes vs. Shares",
        width=450,
        height=300,
    )
)
engagement_chart
