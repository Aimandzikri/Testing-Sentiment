import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="ZM Proshop Insight Dashboard")

# --- NLTK Data Downloads ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Data Loading ---
@st.cache_data
def load_data():
    """
    Loads the mock review dataset.
    """
    data_path = r'C:\Users\aiman\OneDrive\Desktop\PythonProject\ZM_Proshop_Dashboard\data\mock_reviews.csv'
    df = pd.read_csv(data_path)
    return df

# --- Text Preprocessing ---
def preprocess_text(text):
    """
    Cleans and preprocesses the review text.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Sentiment Analysis ---
def analyze_sentiment(text, custom_lexicon):
    """
    Analyzes the sentiment of the text using VADER with a custom lexicon.
    """
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(custom_lexicon)
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# --- Topic Modeling ---
@st.cache_data
def get_topics(text_data, n_topics=3):
    """
    Performs topic modeling using LDA.
    """
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    count_data = vectorizer.fit_transform(text_data)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(count_data)
    
    return lda, vectorizer

# --- Main Dashboard ---
def main():
    """
    The main function for the Streamlit dashboard.
    """
    st.title("⭐ ZM Proshop Customer Feedback Dashboard")
    st.markdown("""
    Welcome to the ZM Proshop Insight Dashboard! 
    This tool helps you understand customer feedback from different platforms.
    Use the filters on the left to explore what customers are saying.
    """)

    # Load data
    df = load_data()

    # Preprocess text
    df['cleaned_comment'] = df['comment'].apply(preprocess_text)

    # Custom Lexicon for more accurate sentiment
    custom_lexicon = {
        "terbaik": 3.0, "mantap": 3.0, "suka": 2.0, "cun": 2.0, "laju": 1.5,
        "pantas": 1.5, "kemas": 1.5, "berbaloi": 2.5, "murah": 1.0, "mahal": -2.0,
        "lambat": -2.0, "tak": -1.5, "bukan": -1.5, "jangan": -2.5, "tipu": -3.0,
        "menyesal": -3.0
    }

    # Analyze sentiment
    df['sentiment_score'] = df['comment'].apply(lambda x: analyze_sentiment(x, custom_lexicon))
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Positive' if x >= 0.05 else ('Neutral' if -0.05 < x < 0.05 else 'Negative'))

    # Get topics
    long_comments = df[df['cleaned_comment'].str.split().str.len() > 3]['cleaned_comment']
    lda, vectorizer = None, None
    if not long_comments.empty:
        try:
            lda, vectorizer = get_topics(long_comments, n_topics=3)
            topic_results = lda.transform(vectorizer.transform(df['cleaned_comment']))
            df['topic'] = [f'Topic {i+1}' for i in topic_results.argmax(axis=1)]
        except Exception as e:
            df['topic'] = 'N/A'
    else:
        df['topic'] = 'N/A'


    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    st.sidebar.markdown("Adjust the filters below to see specific feedback.")
    platform_filter = st.sidebar.multiselect("Platform", df['platform'].unique(), df['platform'].unique())
    sentiment_filter = st.sidebar.multiselect("Sentiment Category", df['sentiment'].unique(), df['sentiment'].unique())

    # Filtered data
    filtered_df = df[(df['platform'].isin(platform_filter)) & (df['sentiment'].isin(sentiment_filter))]

    # --- Main Content ---
    if filtered_df.empty:
        st.warning("No comments match the current filters. Please adjust your selection.")
    else:
        st.header("How do customers feel?")
        with st.expander("What is this?", expanded=False):
            st.write("""
            This section analyzes the emotion behind the comments.
            - The **Average Sentiment Score** gauge shows the overall feeling, from -1 (very negative) to +1 (very positive).
            - The **Sentiment Breakdown** chart shows the number of comments in each category.
            """)

        col1, col2 = st.columns([1, 2])

        with col1:
            # Sentiment Gauge
            avg_sentiment = filtered_df['sentiment_score'].mean()
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_sentiment,
                title = {'text': "Average Sentiment"},
                gauge = {'axis': {'range': [-1, 1], 'tickwidth': 1},
                         'bar': {'color': "darkblue"},
                         'steps' : [
                             {'range': [-1, -0.05], 'color': "#FF4B4B"},
                             {'range': [-0.05, 0.05], 'color': "lightgray"},
                             {'range': [0.05, 1], 'color': "#28A745"}]}))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Sentiment Distribution
            sentiment_counts = filtered_df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'])
            fig_bar = go.Figure(go.Bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                marker_color=['#28A745', 'lightgray', '#FF4B4B'],
                text=sentiment_counts.values,
                textposition='auto'
            ))
            fig_bar.update_layout(
                title_text="Sentiment Breakdown",
                xaxis_title="Sentiment",
                yaxis_title="Number of Comments",
                height=250,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)


        st.header("What are customers talking about?")
        with st.expander("What is this?", expanded=False):
            st.write("""
            This section discovers the main themes from the comments.
            Each **word cloud** represents a key topic. The bigger the word, the more important it is for that topic.
            """)

        if lda and vectorizer:
            # Topic Word Clouds
            feature_names = vectorizer.get_feature_names_out()
            cols = st.columns(3)
            for i, topic in enumerate(lda.components_):
                col = cols[i % 3]
                with col:
                    st.subheader(f"Key Topic {i+1}")
                    top_words = [feature_names[j] for j in topic.argsort()[:-10 - 1:-1]]
                    wordcloud = WordCloud(width=300, height=200, background_color='white', colormap='viridis').generate(' '.join(top_words))
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
        else:
            st.info("Not enough comment data to identify key topics.")


        st.header("Browse Individual Comments")
        st.dataframe(filtered_df[['comment', 'platform', 'sentiment', 'topic']].rename(
            columns={'comment': 'Comment', 'platform': 'Platform', 'sentiment': 'Sentiment', 'topic': 'Inferred Topic'}
        ))

if __name__ == "__main__":
    main()