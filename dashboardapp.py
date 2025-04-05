import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set Streamlit Page Configuration (Must be the first Streamlit command)
st.set_page_config(layout="wide")

# Load JSON Data
with open('customer_analysis_output.json', 'r', encoding='utf-8') as f:
    feedback_data = json.load(f)

# Convert JSON feedback data into a DataFrame for preview
df = pd.DataFrame(feedback_data)

# Function to aggregate emotions
def aggregate_emotions(feedback_list, category):
    emotion_counts = {"Joy": 0, "Surprise": 0, "Trust": 0, "Fear": 0, "Sadness": 0, 
                      "Disgust": 0, "Anger": 0, "Anticipation": 0, "Neutral": 0}
    total_feedbacks = len(feedback_list)

    for feedback in feedback_list:
        emotions = feedback.get("emotion", {})  
        if category not in emotions:
            continue  

        emotion_data = emotions[category]
        emotion_type = emotion_data.get("type", "").capitalize()
        intensity = emotion_data.get("intensity", "").lower()

        if emotion_type in emotion_counts:
            score = 100 if intensity == "high" else 50 if intensity == "medium" else 20 if intensity == "low" else 0
            emotion_counts[emotion_type] += score

    if total_feedbacks > 0:
        for emotion in emotion_counts:
            emotion_counts[emotion] /= total_feedbacks

    return dict(sorted(emotion_counts.items(), key=lambda item: item[1], reverse=True))

# Get aggregated emotion data
primary_emotions = aggregate_emotions(feedback_data, "primary")
secondary_emotions = aggregate_emotions(feedback_data, "secondary")
tertiary_emotions = aggregate_emotions(feedback_data, "tertiary")

# Combine for overall emotion ranking
overall_emotions = primary_emotions.copy()
for emotion, value in secondary_emotions.items():
    overall_emotions[emotion] = overall_emotions.get(emotion, 0) + value
for emotion, value in tertiary_emotions.items():
    overall_emotions[emotion] = overall_emotions.get(emotion, 0) + value
overall_emotions = dict(sorted(overall_emotions.items(), key=lambda item: item[1], reverse=True))

# Extract overall adorescore
overall_adorescore = sum(
    feedback.get("adorescore", {}).get("overall", 50)  # Default 50 if missing
    for feedback in feedback_data
) / len(feedback_data)

# Get the most dominant emotion
dominant_emotion = list(overall_emotions.keys())[0]
driven_by_message = f"Driven by **{dominant_emotion.capitalize()}**-Based Sentiment"

# Display title and Adorescore
left_col, right_col = st.columns([1.5, 1])
with left_col:
    st.markdown("<h2>📌 Customer Emotion Analysis</h2>", unsafe_allow_html=True)
with right_col:
    st.markdown(f"""
        <h2 style="text-align:center;">
        🔥 Overall Adorescore: {overall_adorescore:.2f} 🔥
        </h2>
        <p style="text-align:center; color:#555;">{driven_by_message}</p>
    """, unsafe_allow_html=True)

# Collapsible Dataset Preview
with st.expander("📊 Click to Expand Dataset Preview"):
    st.dataframe(df)

# Emotion Ranking Toggle Buttons
ranking_option = st.radio("Select Emotion Category:", ["Overall", "Primary", "Secondary", "Tertiary"], horizontal=True)

# Set ranking emotions
ranked_emotions = {
    "Overall": overall_emotions,
    "Primary": primary_emotions,
    "Secondary": secondary_emotions,
    "Tertiary": tertiary_emotions
}[ranking_option]

# Display Sorted Emotions as Heatmap
st.markdown("### 📊 Emotion Breakdown")
fig = go.Figure(data=go.Heatmap(
    z=[list(ranked_emotions.values())],
    x=list(ranked_emotions.keys()),
    y=[ranking_option],
    colorscale=[[0, 'rgb(198, 219, 239)'], [1, 'rgb(8, 81, 156)']],
    showscale=True
))
fig.update_layout(xaxis_title="Emotions", yaxis_title="Category", height=400)
st.plotly_chart(fig, use_container_width=True)

# Function to create radar charts
def plot_radar_chart(emotions):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(emotions.values()) + [list(emotions.values())[0]],  
        theta=list(emotions.keys()) + [list(emotions.keys())[0]],
        fill='toself',
        line=dict(color='rgb(8, 81, 156)'),
        fillcolor='rgba(8, 81, 156, 0.3)'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=350, width=350)
    return fig

# Display Emotion Distribution
st.markdown("### 🔄 Emotion Distribution Across Categories")
radar_col1, radar_col2, radar_col3 = st.columns(3)
for col, emotions, title in zip([radar_col1, radar_col2, radar_col3], [primary_emotions, secondary_emotions, tertiary_emotions], ["🌟 Primary Emotions", "🌊 Secondary Emotions", "🎮 Tertiary Emotions"]):
    with col:
        st.markdown(f"### {title}")
        st.plotly_chart(plot_radar_chart(emotions), use_container_width=True)



def extract_topic_words(data):
    words = []
    for item in data:
        for key in ['primary', 'secondary', 'tertiary']:
            if key in item.get('topics', {}):
                words.append(item['topics'][key]['topic'])
        subtopics = item.get('topics', {}).get('subtopics', {})
        for sublist in subtopics.values():
            words.extend(sublist)
        for sentence in item.get('adorescore', {}).get('sentence_analysis', []):
            words.extend(sentence.get('topics', {}).keys())
    return words

# Extract topic distribution
topics_list = extract_topic_words(feedback_data)
topic_counts = Counter(topics_list)

# Prepare data for visualization
topic_labels = list(topic_counts.keys())
topic_values = list(topic_counts.values())

# Create parents array (All topics belong to "Topics")
topic_parents = ["Topics"] * len(topic_labels)





# **Treemap Chart for Topics**
st.markdown("### 🌳 Treemap of Topics")
fig_treemap = go.Figure(go.Treemap(
    labels=["Topics"] + topic_labels,
    parents=[""] + topic_parents,
    values=[sum(topic_values)] + topic_values
))
fig_treemap.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
st.plotly_chart(fig_treemap, use_container_width=True)

# **Word Cloud for Topics**
if len(topic_labels) < 3:
    st.warning("⚠️ Not enough data to generate a word cloud. Please check the dataset.")
else:
    wordcloud_text = " ".join(topic_labels)
    wordcloud = WordCloud(width=800, height=400, background_color="white", min_font_size=1).generate(wordcloud_text)
    st.markdown("### ☁️ Frequent Topics Word Cloud")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
# Generate Word Cloud
word_list = extract_topic_words(feedback_data)
if len(word_list) < 3:
    st.warning("⚠️ Not enough data to generate a word cloud. Please check the dataset.")
else:
    wordcloud_text = " ".join(word_list)
    wordcloud = WordCloud(width=800, height=400, background_color="white", min_font_size=1).generate(wordcloud_text)
    st.markdown("### ☁️ Frequent Topics Word Cloud")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# Display Top 5 Themes
st.markdown("### 🔥 Top 5 Dominant Themes")

# Create DataFrame and group by Topic to sum occurrences
topic_data = pd.DataFrame({"Topic": topic_labels, "Count": topic_values})
top_5_themes = topic_data.groupby("Topic", as_index=False)["Count"].sum()

# Sort and select top 5
top_5_themes = top_5_themes.sort_values(by="Count", ascending=False).head(5)

# Display in Streamlit
st.dataframe(top_5_themes.set_index("Topic"))


# Advanced Filter Options
st.markdown("### 🎯 Advanced Filters")

# Extract unique topics and emotions from the data
all_topics = set()
all_emotions = set()

for feedback in feedback_data:
    topics = feedback.get("topics", {}).get("subtopics", {}).keys()
    emotions = feedback.get("emotion", {}).values()
    
    all_topics.update(topics)
    all_emotions.update(e.get("type", "").capitalize() for e in emotions if isinstance(e, dict))

# Sort options for better UI experience
all_topics = sorted(all_topics)
all_emotions = sorted(all_emotions)

# Dropdown for Topic Selection
selected_topic = st.selectbox("📌 Select Topic", ["All"] + all_topics, index=0)

# Dropdown for Emotion Selection
selected_emotion = st.selectbox("😊 Select Emotion", ["All"] + all_emotions, index=0)

# Dropdown for Emotion Intensity Selection
selected_intensity = st.selectbox("🔥 Select Intensity Level", ["All", "Low", "Medium", "High"], index=0)

# Function to filter feedback based on selected options
def filter_feedback(data, topic=None, emotion=None, intensity=None):
    filtered_results = []
    
    for entry in data:
        entry_topics = entry.get("topics", {}).get("subtopics", {})
        entry_emotions = entry.get("emotion", {})
        
        # Check if topic matches
        if topic and topic != "All" and topic not in entry_topics:
            continue
        
        # Check if emotion matches
        matched_emotion = False
        for e in entry_emotions.values():
            if isinstance(e, dict) and e.get("type", "").capitalize() == emotion:
                if intensity == "All" or e.get("intensity", "").lower() == intensity.lower():
                    matched_emotion = True
                    break
        if emotion != "All" and not matched_emotion:
            continue

        # Retrieve matching subtopics
        subtopics = entry_topics.get(topic, []) if topic and topic != "All" else list(entry_topics.keys())

        filtered_results.append({
            "feedback_text": entry["feedback_text"],
            "emotion": e.get("type", ""),
            "intensity": e.get("intensity", ""),
            "topics": subtopics
        })
    
    return filtered_results

# Apply filter function
filtered_feedback = filter_feedback(feedback_data, selected_topic, selected_emotion, selected_intensity)

# Display filtered results
st.markdown(f"### 🎯 Filtered Results ({len(filtered_feedback)} Matches)")
if filtered_feedback:
    st.dataframe(pd.DataFrame(filtered_feedback))
else:
    st.warning("No matching feedback found. Try adjusting the filters.")

































import streamlit as st
import json
import spacy
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob
from keybert import KeyBERT

# Load Spacy Model
nlp = spacy.load("en_core_web_sm")

# Initialize KeyBERT Model
kw_model = KeyBERT()

# Emojis for Better UI
EMOTIONS_MAP = {
    "joy": "😊 Joy",
    "sadness": "😢 Sadness",
    "anger": "😡 Anger",
    "fear": "😨 Fear",
    "surprise": "😲 Surprise",
    "neutral": "😐 Neutral"
}

# Function to Extract Topics & Subtopics
def extract_topics(text, top_n=5):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    topics = [kw[0] for kw in keywords]

    # Extracting subtopics using noun chunks
    doc = nlp(text)
    subtopics = [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() not in topics]

    return topics, subtopics

# Function for Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    sentiment_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return {"polarity": round(polarity, 2), "subjectivity": round(subjectivity, 2), "sentiment": sentiment_label}

# Function to Detect Emotions
def analyze_emotion(text):
    emotion_words = {
        "joy": ["happy", "joyful", "excited", "pleased"],
        "sadness": ["sad", "depressed", "unhappy", "gloomy"],
        "anger": ["angry", "furious", "mad", "annoyed"],
        "fear": ["afraid", "scared", "frightened", "nervous"],
        "surprise": ["surprised", "shocked", "astonished"],
    }

    doc = nlp(text.lower())
    detected_emotions = {emotion: sum(1 for token in doc if token.text in words) for emotion, words in emotion_words.items()}
    
    detected_emotions = {k: v for k, v in detected_emotions.items() if v > 0}
    primary_emotion = max(detected_emotions, key=detected_emotions.get, default="neutral")

    return {"primary_emotion": EMOTIONS_MAP[primary_emotion], "emotions_detected": detected_emotions}

# Function to Generate Word Cloud (Reduced Size)
def generate_wordcloud(text):
    wordcloud = WordCloud(width=600, height=300, background_color="white").generate(text)
    plt.figure(figsize=(6, 3))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Function to Generate Topic Pie Chart
def generate_pie_chart(topics):
    topic_counts = Counter(topics)
    if topic_counts:
        fig = px.pie(
            names=list(topic_counts.keys()),
            values=list(topic_counts.values()),
            title="🍕 Topic Distribution",
            hole=0.4
        )
        st.plotly_chart(fig)

# Function to Calculate Adorescore
def calculate_adorescore(sentiment, topic_count):
    sentiment_score = (sentiment["polarity"] + 1) * 5  # Scale polarity to 0-10
    topic_score = min(topic_count * 2, 10)  # Scale topic count to max 10
    adorescore = (0.7 * sentiment_score) + (0.3 * topic_score)  # Weighted Score

    return round(adorescore, 2)

# Streamlit UI
st.title("📝 AI-Powered Customer Feedback Analysis")

# User Input
user_input = st.text_area("💬 Enter Customer Feedback:")

if st.button("🚀 Analyze"):
    if user_input:
        # Extract Topics & Subtopics
        topics, subtopics = extract_topics(user_input)

        # Sentiment Analysis
        sentiment_result = analyze_sentiment(user_input)

        # Emotion Detection
        emotion_result = analyze_emotion(user_input)

        # Calculate Adorescore
        adorescore = calculate_adorescore(sentiment_result, len(topics))

        # Final JSON Output
        result_data = {
            "emotion": emotion_result["primary_emotion"],
            "topics": topics,
            "subtopics": subtopics,
            "adorescore": adorescore
        }

        # Display JSON output directly
        st.subheader("📜 JSON Output")
        st.json(result_data)

        # Visualization
        generate_pie_chart(topics)
        generate_wordcloud(user_input)

    else:
        st.warning("⚠️ Please enter feedback for analysis!")
