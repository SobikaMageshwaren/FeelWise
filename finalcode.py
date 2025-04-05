import os
import json
import warnings
import spacy
import nltk
import re
from transformers import pipeline
from deep_translator import GoogleTranslator
from keybert import KeyBERT
from nltk.sentiment import SentimentIntensityAnalyzer

# to remove warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Load the pre- trained DistilRoBERTa model(for emotion classification)
emotionmodel = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
#add a googletranslator funciton to translate non-english feedback, for  multilingual support
translator = GoogleTranslator()
#to parse text,tokenize,perform named entity recognition like product names,locations, classifying the words i.e part of speech. and have a clear grammatical structure and relationships
nlp = spacy.load("en_core_web_sm")

#use the model for keyword extraction to find the key topics, relevant words and categorizing hte feedback
keyword_model = KeyBERT()

#to analyze the tone of customer feedback and calculate the adorescore
sa= SentimentIntensityAnalyzer()

#to assign topics dynamically
topic_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Give the json file path(result path)
JSON_FILE = "customer_analysis_output.json"

#to translate text

def totranslatetext(text):
    try:
        return translator.translate(text, target="en")
    #to handle exceptions
    except Exception:
        return text


def todetectemotion(text):  # Function to detect emotion from user input text
    try:
        # Pass the input text to the emotion model for classification
        results = emotionmodel(text)

        # Sort the detected emotions by confidence score in descending order
        sorted_emotions = sorted(results[0], key=lambda x: x["score"], reverse=True)

        def get_emotion_details(index):  # Function to extract details of a specific emotion
            if index < len(sorted_emotions):  # Check if the requested index is within range
                emotion = sorted_emotions[index]  # Get the emotion at the given index
                confidence = round(emotion["score"], 2)  # Round off the confidence score to 2 decimal places
                
                # Determine the intensity level based on confidence score
                if confidence <= 0.3:
                    intensity = "low"
                elif confidence <= 0.7:
                    intensity = "medium"
                else:
                    intensity = "high"

                # Return a dictionary containing emotion type, intensity, and confidence score
                return {"type": emotion["label"], "intensity": intensity, "confidence": confidence}
            
            return None  # Return None if the index is out of range (fewer detected emotions)

        # Return a dictionary containing the top three detected emotions with their details
        return {
            "primary": get_emotion_details(0),   # Most dominant emotion
            "secondary": get_emotion_details(1), # Second most dominant emotion
            "tertiary": get_emotion_details(2)   # Third most dominant emotion 
        }
    
    except Exception:  # Handle any potential errors (e.g., model failure, empty input)
        return None  # Return None in case of an error


# Function to extract key topics from the given text
def extracttopics(text):
    # Use the keyword extraction model to identify important words or phrases
    # - keyphrase_ngram_range=(1,2) allows extraction of both single words and two-word phrases
    # - stop_words="english" removes common words like "the", "and", etc.
    # - top_n=10 extracts the top 10 most relevant keywords
    key_topics = keyword_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10)

    # Process the text using an NLP model to extract named entities (e.g., people, places, organizations)
    doc = nlp(text)
    
    # Convert named entities into a unique list to avoid duplicates
    named_entities = list(set(ent.text for ent in doc.ents))

    # Combine extracted keywords with named entities, assigning a confidence score of 1.0 to named entities
    main_topics = key_topics + [(entity, 1.0) for entity in named_entities]

    # Sort topics in descending order based on confidence scores (higher confidence comes first)
    sorted_topics = sorted(main_topics, key=lambda x: x[1], reverse=True)

    # Function to retrieve topic details at a given index
    def togettopicdetails(index):
        if index < len(sorted_topics):  # Ensure the index is within bounds
            return {"topic": sorted_topics[index][0], "confidence": sorted_topics[index][1]}  # Return topic and confidence
        return None  # Return None if there are not enough topics

    # Create a dictionary with primary, secondary, and tertiary topics along with their confidence scores
    return {
        "primary": togettopicdetails(0),   # Most relevant topic
        "secondary": togettopicdetails(1), # Second most relevant topic
        "tertiary": togettopicdetails(2),  # Third most relevant topic

        # Identify related subtopics by checking which extracted keywords contain each top topic
        # This groups related keywords under the top 3 topics
        "subtopics": {t[0]: [kw[0] for kw in key_topics if kw[0] in t[0]] for t in sorted_topics[:3]}
    }


#To  normalize Sentiment Score
def normalizescore(score, min_input=-1, max_input=1, min_output=0, max_output=100):
    return ((score - min_input) / (max_input - min_input)) * (max_output - min_output) + min_output

# Extract the required sentences
def toextractsentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

# Function to analyze the given text for sentiment and extract relevant topics
def analyzetext(text):
    # Break the input text into individual sentences for analysis
    sentences = toextractsentences(text)
    
    # List to store sentiment and topic analysis results for each sentence
    sentenceresults = []
    
    # Dictionary to keep track of topic relevance scores across all sentences
    topicrelevance = {}

    # Process each sentence one by one
    for sentence in sentences:
        # Skip empty sentences (in case there are extra spaces or line breaks)
        if not sentence.strip():
            continue

        # Perform sentiment analysis on the sentence using the sentiment analyzer (sa)
        sentiment = sa.polarity_scores(sentence)
        
        # Extract the overall sentiment score (compound score ranges from -1 to 1)
        sentiment_score = sentiment['compound']
        
        # Normalize the sentiment score to make it more interpretable
        normalized_sentiment = normalizescore(sentiment_score)

        # Extract key topics from the sentence
        # - keyphrase_ngram_range=(1,2) allows single words and two-word phrases
        # - stop_words="english" removes common words
        # - top_n=5 ensures we extract only the most important topics
        extracted_topics = keyword_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5)

        # Convert extracted topics into a dictionary with scores scaled to a percentage (0-100)
        detected_topics = {topic: round(score * 100, 2) for topic, score in extracted_topics}

        # Aggregate topic relevance scores across all sentences
        for topic, relevance in detected_topics.items():
            topicrelevance.setdefault(topic, []).append(relevance)

        # Store the results for the sentence, including:
        # - The original sentence
        # - The rounded normalized sentiment score
        # - The detected topics and their relevance scores
        sentenceresults.append({
            "sentence": sentence,
            "sentiment_score": round(normalized_sentiment, 2),
            "topics": detected_topics
        })

    # Return:
    # - A list of analyzed sentences with their sentiment and topics
    # - A dictionary tracking the relevance of each topic across multiple sentences
    return sentenceresults, topicrelevance


# Function to compute the overall Adorescore based on text analysis
def calculateadorescore(text):
    # Step 1: Analyze the text to extract sentence-wise sentiment and topic relevance
    sentence_results, topic_relevance = analyzetext(text)

    # Step 2: Compute the average relevance score for each detected topic
    # - For each topic, sum all relevance scores from different sentences
    # - Divide by the number of occurrences to get the average score
    # - Round the final value to 2 decimal places for readability
    topic_scores = {topic: round(sum(scores) / len(scores), 2) for topic, scores in topic_relevance.items() if scores}

    # Step 3: Compute the overall Adorescore
    # - Take the average of all topic scores
    # - If no topics were detected, default the Adorescore to 50 (neutral value)
    overall_adorescore = round(sum(topic_scores.values()) / len(topic_scores), 2) if topic_scores else 50

    # Step 4: Return the results as a structured dictionary
    # - "sentence_analysis": Contains sentiment and topic details for each sentence
    # - "adorescore":
    #   - "overall": The final computed Adorescore
    #   - "breakdown": Topic-wise contribution to the Adorescore
    return {
        "sentence_analysis": sentence_results,
        "adorescore": {
            "overall": overall_adorescore,
            "breakdown": topic_scores
        }
    }

# Function to process user feedback and extract meaningful insights
def processingfeedback(feedback_text):
    # Step 1: Translate the feedback text (if needed)
    # - This ensures we can analyze feedback in a common language, improving accuracy
    translated_text = totranslatetext(feedback_text)

    # Step 2: Detect the emotions expressed in the feedback
    # - This helps understand the user's sentiment and intensity of emotions
    emotion = todetectemotion(translated_text)

    # Step 3: Extract key topics from the feedback
    # - Identifies the main themes discussed in the user's input
    topics = extracttopics(translated_text)

    # Step 4: Compute the Adorescore
    # - Evaluates the overall sentiment and topic relevance in a structured way
    adorescore = calculateadorescore(translated_text)

    # Step 5: Return a structured response with all processed data
    # - "feedback_text": Original user feedback
    # - "translated_text": Feedback after translation (if any)
    # - "emotion": Detected emotions with intensity and confidence
    # - "topics": Primary, secondary, tertiary topics + related subtopics
    # - "adorescore": Overall sentiment score and topic breakdown
    return {
        "feedback_text": feedback_text,
        "translated_text": translated_text,
        "emotion": emotion,
        "topics": {
            "primary": topics["primary"],
            "secondary": topics["secondary"],
            "tertiary": topics["tertiary"],
            "subtopics": topics["subtopics"]
        },
        "adorescore": adorescore
    }


# To continuously store the input feedback in the json file
def saveoutputtojsonformat(new_data, file_path=JSON_FILE):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                stored_data= json.load(f)
                if not isinstance(stored_data, list):
                    stored_data = [stored_data]
            except json.JSONDecodeError:
                stored_data = []
    else:
        stored_data = []
    
    stored_data.append(new_data)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(stored_data, f, ensure_ascii=False, indent=4)

# to get the user feedback
userinputfeedback = input("Enter customer feedback: ")
finalres = processingfeedback(userinputfeedback)

# At the end, save the result
saveoutputtojsonformat(finalres)

print("\n Customer feedback is analyzed and stored in 'customer_analysis_output.json'")

