import os
import pandas as pd
import openai
from bertopic import BERTopic
from bertopic.representation import OpenAI as BEROpenAI
from sklearn.feature_extraction.text import CountVectorizer

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

# Load your data from a JSON file and sample 1000 records
try:
    df = pd.read_json("data/36f447ca-9895-422b-a9ff-5c0516513f95AviationData.json", lines=True)
except ValueError:
    df = pd.read_json("data/36f447ca-9895-422b-a9ff-5c0516513f95AviationData.json")
df = df.sample(1000)

# Extract the 'ProbableCause' column, dropping any missing values
docs = df["ProbableCause"].dropna().tolist()

# Define your custom prompt
custom_prompt = """
You are an expert aviation accident analyst. Each topic is summarized by:
- KEYWORDS: [KEYWORDS]
- SAMPLE DOCUMENTS: [DOCUMENTS]

Your task:
1. Carefully review the keywords and sample documents.
2. Identify the dominant contributing factor for the accidents described.
3. Choose exactly one label from the following options:
   - Pilot Error
   - Student Pilot
   - Mechanical Issues
   - Runway Issues
   - Environmental Factors
   - Flight Training Error
   - Other

Guidelines:
- If the text indicates that the accident is related to inadequate pilot actions *and* shows evidence of delayed or inadequate remedial action from the flight instructor, label the topic as **Flight Training Error**.
- If the issue is solely due to pilot error, use **Pilot Error** or **Student Pilot** as appropriate.
- For cases where contamination or part failure is the main issue, label as **Mechanical Issues**.
- Do not mix more than one label; choose the one that best represents the dominant contributing factor.
- Keep the label short (3-4 words maximum) with no extra explanation.

Output format:
topic: <label>
"""

# Create the OpenAI client and wrap it with BERTopic's OpenAI representation model
client = openai.OpenAI(api_key=openai.api_key)
representation_model = BEROpenAI(
    client,
    prompt=custom_prompt,
    nr_docs=10,          # number of sample documents per topic to include in the prompt
    delay_in_seconds=2   # optional delay between API calls
)

# Create a custom vectorizer to capture technical phrases (using n-grams from 1 to 10 words)
vectorizer_model = CountVectorizer(ngram_range=(1, 10), stop_words="english")

# Instantiate BERTopic with both the custom representation model and custom vectorizer
topic_model = BERTopic(
    representation_model=representation_model,
    vectorizer_model=vectorizer_model,
    language="english",
    verbose=True
)

# Fit the topic model on your documents to get topics and probabilities
topics, probs = topic_model.fit_transform(docs)

# Inspect the initial topic information
initial_topic_info = topic_model.get_topic_info()
print("Initial Topic Information:")
print(initial_topic_info.head())

# Reduce topics to merge similar ones (targeting 10 topics in this example)
reduced_topics, reduced_probs = topic_model.reduce_topics(docs, topics, nr_topics=10)
reduced_topic_info = topic_model.get_topic_info()
print("Reduced Topic Information:")
print(reduced_topic_info.head())

# Map the (reduced) topic labels back to the original DataFrame
filtered = df[df["ProbableCause"].notna()].reset_index(drop=True)
filtered["TopicID"] = reduced_topics
filtered["TopicName"] = filtered["TopicID"].map(dict(zip(reduced_topic_info.Topic, reduced_topic_info.Name)))

print("Sample of Labeled Documents:")
print(filtered[["ProbableCause", "TopicID", "TopicName"]].head(200))

# Save the processed DataFrame to a CSV file
filtered.to_csv("data/processed_aviation_data.csv", index=False)
