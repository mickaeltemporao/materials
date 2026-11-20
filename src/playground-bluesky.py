"""
Bluesky Data Collection & Text Analysis
"""

import requests
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Fetch Posts from Bluesky Public API (No API key needed)
# -----------------------------------------------------------------------------
# Note: Emmanuel Macron does not have an official personal Bluesky account.
# We use the party/majority account (or 'info.gouv.fr', 'lemonde.fr', 'afp.com'):
handle = "avecrenaissance.bsky.social"
url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={handle}&limit=50"

data = requests.get(url).json()

# -----------------------------------------------------------------------------
# 2. Extract Text, Likes, and Shares
# -----------------------------------------------------------------------------
posts = []
for item in data.get("feed", []):
    post = item["post"]
    posts.append({
        "text": post["record"]["text"],
        "likes": post.get("likeCount", 0),
        "shares": post.get("repostCount", 0),
    })

# -----------------------------------------------------------------------------
# 3. Save to File & Load into a DataFrame
# -----------------------------------------------------------------------------
# Save to CSV
pd.DataFrame(posts).to_csv("bluesky_posts.csv", index=False)

# Load from CSV (interactive REPL workflow)
df = pd.read_csv("bluesky_posts.csv")
df.head()

# -----------------------------------------------------------------------------
# 4. Count Occurrences of a Keyword
# -----------------------------------------------------------------------------
# Number of posts mentioning "étudiant" (case-insensitive):
df["text"].str.contains("étudiant", case=False, na=False).sum()

# View only the posts that mention it:
df[df["text"].str.contains("étudiant", case=False, na=False)]
