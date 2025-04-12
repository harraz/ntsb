import os
import pandas as pd
import openai
from bertopic import BERTopic
from bertopic.representation import OpenAI as BEROpenAI
from sklearn.feature_extraction.text import CountVectorizer

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

# Load your data (with low_memory=False to avoid dtype warnings)
df = pd.read_csv(
    "data/c98ea656-d10e-42e1-ba6b-1ec66280a6c0AviationData.csv",
    low_memory=False
).sample(10000)
docs = df["ProbableCause"].dropna().tolist()

# Define a refined custom prompt to explicitly distinguish runway issues
custom_prompt = """
You are analyzing aviation accident topics. Each topic is described by:
- A set of KEYWORDS: [KEYWORDS]
- A few SAMPLE DOCUMENTS: [DOCUMENTS]

Your goal:
1. Carefully read the keywords and sample documents.
2. Identify all significant factors mentioned, such as mechanical issues, runway conditions, environmental factors, pilot actions, fuel management, and visual/lighting conditions.
3. Provide a SINGLE, concise label that reflects the dominant contributing factor(s).

Important:
- If the reports mention runway misalignment, inadequate runway lighting, or a lack of visual cues (e.g., darkness, insufficient runway lights), label the topic to reflect those issues, for example, "Runway Error" or "Poor Runway Lighting".
- Only label as "Fuel Management" if the main issue is related to fuel exhaustion, refueling errors, or inadequate fuel planning.
- Do NOT include "engine power" if the primary issue is about runway or lighting factors.
- Keep the label short (2-3 words) with no extra explanation.
- Do not repeat the same keywords or sample documents in your label.

Output format:
topic: <label>
"""

# Create the OpenAI client
client = openai.OpenAI(api_key=openai.api_key)

# Wrap the client with BERTopic's OpenAI representation model, passing your custom prompt
representation_model = BEROpenAI(
    client,
    prompt=custom_prompt,
    nr_docs=50,          # number of sample docs per topic to include in the prompt
    delay_in_seconds=2   # optional delay between API calls to avoid rate limits
)

# Create a custom vectorizer with desired n-gram range to capture technical phrases
vectorizer_model = CountVectorizer(ngram_range=(1, 4), stop_words="english")

# Instantiate BERTopic with both the custom representation model and custom vectorizer
topic_model = BERTopic(
    representation_model=representation_model,
    vectorizer_model=vectorizer_model,
    language="english",
    verbose=True
)

# Fit the model & generate topic labels
topics, probs = topic_model.fit_transform(docs)

# Inspect the auto‑generated topic names
topic_info = topic_model.get_topic_info()
print(topic_info.head())

# (Optional) Map topic labels back to your original DataFrame
filtered = df[df["ProbableCause"].notna()].reset_index(drop=True)
filtered["TopicID"]   = topics
filtered["TopicName"] = filtered["TopicID"].map(
    dict(zip(topic_info.Topic, topic_info.Name))
)
print(filtered[["ProbableCause", "TopicID", "TopicName"]].head())
filtered.to_csv("data/processed_aviation_data.csv", index=False)
