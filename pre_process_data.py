from bertopic import BERTopic
import pandas as pd

# 1) Load your data
df = pd.read_csv("data/c98ea656-d10e-42e1-ba6b-1ec66280a6c0AviationData.csv",  low_memory=False).sample(1000)

# 2) Pull out the non‑null causes
docs = df["ProbableCause"].dropna().tolist()

# 3) Instantiate BERTopic
#    - language="english" uses English stop‑words
#    - nr_topics="auto" will merge/split to find a good number of topics
topic_model = BERTopic(language="english", nr_topics="auto")

# 4) Fit the model & get back (topic_id, probability) for each doc
topics, probs = topic_model.fit_transform(docs)

# 5) Get a DataFrame of topics (with automatically generated names & sizes)
topic_info = topic_model.get_topic_info()
print(topic_info.head())
# ┌────────┬─────────────────────────┬───────┐
# │ Topic  │ Name                    │ Count │
# ├────────┼─────────────────────────┼───────┤
# │ -1     │ -1_Other                │   123 │   ← outliers
# │ 0      │ fuel_exhaustion_power   │   456 │
# │ 1      │ landing_gear_failure    │   389 │
# │ 2      │ throttle_cable_separation│  272 │
# │ …      │ …                       │   …   │
# └────────┴─────────────────────────┴───────┘

# 6) Map back to your original DataFrame
#    Note: we dropped NA above, so re‑align carefully
filtered_df = df[df["ProbableCause"].notna()].reset_index(drop=True)
filtered_df["TopicID"]   = topics
filtered_df["TopicName"] = filtered_df["TopicID"].map(
    dict(zip(topic_info.Topic, topic_info.Name))
)

# 7) Inspect your newly labeled data
print(filtered_df[["ProbableCause", "TopicID", "TopicName"]].head(10))
# filtered_df.to_csv("data/processed_aviation_data.csv", index=False)