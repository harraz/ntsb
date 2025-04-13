import os
import pandas as pd
import openai
from bertopic import BERTopic
from bertopic.representation import OpenAI as BEROpenAI
from sklearn.feature_extraction.text import CountVectorizer

def load_data(json_file_path, sample_size=1000):
    """
    Loads data from a JSON file and returns a sampled DataFrame and a list of documents from the 'ProbableCause' field.
    """
    # Adjust the loading method if your JSON is in "json lines" format by setting lines=True.
    try:
        df = pd.read_json(json_file_path, lines=True)
    except ValueError:
        df = pd.read_json(json_file_path)
        
    df = df.sample(sample_size)
    docs = df["ProbableCause"].dropna().tolist()
    return df, docs

def build_representation_model(custom_prompt, nr_docs=10, delay=2):
    """
    Creates and returns a BERTopic OpenAI representation model using the custom prompt.
    """
    # Ensure API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable")
    
    openai.api_key = api_key
    client = openai.OpenAI(api_key=api_key)
    representation_model = BEROpenAI(
        client,
        prompt=custom_prompt,
        nr_docs=nr_docs,
        delay_in_seconds=delay
    )
    return representation_model

def build_vectorizer(ngram_range=(1, 10)):
    """
    Creates and returns a custom CountVectorizer with the specified ngram range and English stop words.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    return vectorizer

def main():
    # Load your JSON data
    json_file = "data/36f447ca-9895-422b-a9ff-5c0516513f95AviationData.json"
    df, docs = load_data(json_file, sample_size=1000)
    
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
    
    # Build the OpenAI representation model with the custom prompt
    representation_model = build_representation_model(custom_prompt, nr_docs=10, delay=2)
    
    # Build a custom vectorizer to capture technical phrases (using a wide ngram range)
    vectorizer_model = build_vectorizer(ngram_range=(1, 10))
    
    # Instantiate BERTopic with both the custom vectorizer and the OpenAI representation model
    topic_model = BERTopic(
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        language="english",
        verbose=True
    )
    
    # Fit the model on the documents to get topics and probabilities
    topics, probs = topic_model.fit_transform(docs)
    
    # Retrieve topic information and display initial topics
    topic_info = topic_model.get_topic_info()
    print("Initial Topic Information:")
    print(topic_info.head())
    
    # Map topic labels back onto the original DataFrame
    filtered = df[df["ProbableCause"].notna()].reset_index(drop=True)
    filtered["TopicID"] = topics
    filtered["TopicName"] = filtered["TopicID"].map(
        dict(zip(topic_info.Topic, topic_info.Name))
    )
    
    print("Sample of Labeled Documents:")
    print(filtered[["ProbableCause", "TopicID", "TopicName"]].head(200))
    
    # Save the processed DataFrame to a CSV file
    filtered.to_csv("data/processed_aviation_data.csv", index=False)

if __name__ == "__main__":
    main()
