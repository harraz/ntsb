import os
import pandas as pd
import openai
from bertopic import BERTopic
from bertopic.representation import OpenAI as BEROpenAI

# 2) Set your OpenAI API key (or export OPENAI_API_KEY in your shell)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Set the OPENAI_API_KEY environment variable")


# 3) Load your data
df = pd.read_csv("data/c98ea656-d10e-42e1-ba6b-1ec66280a6c0AviationData.csv", low_memory=False).sample(1000)
docs = df["ProbableCause"].dropna().tolist()

# 4) Wrap the OpenAI client for BERTopic
#    This uses BERTopic’s default prompt to ask OpenAI for a short label per topic.
client = openai.OpenAI()            # or use openai.ChatCompletion directly
representation_model = BEROpenAI(client)

# 5) Instantiate BERTopic with OpenAI-based representation
topic_model = BERTopic(
    representation_model=representation_model,
    language="english",
    verbose=True
)

# 6) Fit the model & generate topic labels
topics, probs = topic_model.fit_transform(docs)

# 7) Inspect the auto‑generated topic names
topic_info = topic_model.get_topic_info()
print(topic_info.head())

# 8) (Optional) Map back to your original DataFrame
filtered = df[df["ProbableCause"].notna()].reset_index(drop=True)
filtered["TopicID"]   = topics
filtered["TopicName"] = filtered["TopicID"].map(
    dict(zip(topic_info.Topic, topic_info.Name))
)

print(filtered[["ProbableCause", "TopicID", "TopicName"]].head())
filtered.to_csv("data/processed_aviation_data.csv", index=False)