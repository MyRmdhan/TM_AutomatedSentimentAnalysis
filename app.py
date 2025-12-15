import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import re # For regex to extract video ID
import string # For punctuation

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="📺",
    layout="wide"
)

st.title("📺 YouTube Sentiment Analysis")
st.markdown("""
    Aplikasi ini menganalisis sentimen dari komentar video YouTube menggunakan model bahasa IndoBERT.
    Masukkan URL video YouTube di bawah ini untuk memulai analisis.
""")

# --- YouTube API Key Warning ---
st.warning("""
    **Peringatan:** Aplikasi ini memerlukan kunci API YouTube Data API v3.
    Pastikan Anda telah menyetel kunci API sebagai variabel lingkungan `YOUTUBE_API_KEY`.
    Jika belum, silakan ikuti petunjuk di [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
    untuk mendapatkan dan mengaktifkan YouTube Data API v3.
""")

# --- Helper Functions ---

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

# Function to fetch comments from YouTube
@st.cache_data(show_spinner=True)
def get_youtube_comments(video_id, api_key, max_results=100):
    comments_list = []
    youtube = build('youtube', 'v3', developerKey=api_key)

    try:
        # Get video details
        video_response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()

        if not video_response['items']:
            st.error("Video tidak ditemukan atau tidak tersedia.")
            return None, None, None

        video_title = video_response['items'][0]['snippet']['title']
        video_thumbnail = video_response['items'][0]['snippet']['thumbnails']['medium']['url']

        # Get comments
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100 # Initial request
        )
        
        while request and len(comments_list) < max_results:
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments_list.append(comment)
                if len(comments_list) >= max_results:
                    break
            request = youtube.commentThreads().list_next(request, response)

    except Exception as e:
        st.error(f"Error fetching YouTube comments or video details: {e}")
        st.error("Pastikan Video ID atau Kunci API valid dan Anda memiliki akses.")
        return None, None, None

    return comments_list, video_title, video_thumbnail

# Function to load and use IndoBERT sentiment pipeline
@st.cache_resource(show_spinner=True)
def load_sentiment_pipeline():
    model_name = "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True)
    return sentiment_classifier

def clean_text_for_wordcloud(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

def generate_word_cloud(text_data, sentiment_type):
    # Define Indonesian stopwords (can be expanded)
    indo_stopwords = [
        "yang", "untuk", "ini", "itu", "dan", "dengan", "saya", "kamu", "mereka", "kita", "ada",
        "tidak", "bukan", "akan", "sudah", "belum", "sangat", "lebih", "dari", "ke", "di", "pada",
        "juga", "atau", "namun", "tetapi", "serta", "saat", "jika", "maka", "agar", "supaya", "pun",
        "nya", "lah", "kok", "sih", "dong", "deh", "loh", "kan", "ya", "ga", "gak", "udah",
        "yg", "aja", "buat", "pasti", "bgt", "bngt", " banget", "bgt", "nih", "sbnrnya", "tapi", "cuma",
        "jd", "kalo", "udah", "dr", "tp", "pas", "emang", "si", "jd", "udah", "pake", "kek", "dg",
        "jg", "gw", "lu", "kalian", "mereka", "kita", "nya", "gitu", "gak", "jg", "deh", "dari", "yang",
        "ini", "itu", "dan", "dengan", "adalah", "untuk", "pada", "tidak", "ada", "saya", "bisa", "itu",
        "kalau", "mau", "udah", "lagi", "pun", "akan", "saja", "lebih", "baik", "nya", "juga", "sih",
        "jangan", "tapi", "banyak", "seperti", "semua", "bahkan", "jadi", "hanya", "apa", "siapa", "dimana",
        "kapan", "mengapa", "bagaimana", "setelah", "sebelum", "tanpa", "selalu", "sering", "jarang",
        "kadang", "pernah", "tidak pernah", "belum pernah", "masih", "sudah", "tetap", "pasti", "memang",
        "mungkin", "harus", "boleh", "dapat", "perlu", "ingin", "coba", "pasti", "betul", "benar", "salah",
        "kurang", "sedikit", "banyak", "semua", "setiap", "beberapa", "seluruh", "agak", "cukup", "terlalu",
        "sangat", "amat", "begitu", "sekali", "cukup", "kurang", "terlalu", "paling", "sering", "jarang",
        "lama", "baru", "kecil", "besar", "panjang", "pendek", "tinggi", "rendah", "jauh", "dekat",
        "depan", "belakang", "atas", "bawah", "samping", "tengah", "luar", "dalam", "sana", "sini", "situ",
        "mana", "kini", "nanti", "kemarin", "besok", "dulu", "sekarang", "lalu", "kemudian", "segera",
        "nanti", "sejak", "hingga", "sampai", "walaupun", "meskipun", "padahal", "sedangkan", "yaitu",
        "yakni", "artinya", "misalnya", "antara", "lain", "demikian", "begitulah", "karena", "sebab",
        "oleh", "itu", "maka", "dengan", "dan", "atau", "tapi", "tetapi", "serta", "kecuali", "melainkan",
        "bahkan", "lagipula", "apalagi", "padahal", "seandainya", "umpamanya", "misalnya", "jikalau",
        "walaupun", "meskipun", "sungguhpun", "andaikata", "biarpun", "sekalipun", "jika", "kalau"
    ]
    
    # Combine with default English stopwords (if any English comments slip through or for robustness)
    stopwords = set(STOPWORDS)
    stopwords.update(indo_stopwords)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        min_font_size=10
    ).generate(text_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Kata Paling Dominan ({sentiment_type.capitalize()})")
    st.pyplot(fig)

# --- Main Application Logic ---
youtube_url = st.text_input("Masukkan URL Video YouTube:", "https://www.youtube.com/watch?v=dQw4w9WgXcQ") # Default for testing

if st.button("Mulai Analisis Sentimen"):
    if not youtube_url:
        st.error("Mohon masukkan URL video YouTube.")
    else:
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if not youtube_api_key:
            st.error("Kunci API YouTube tidak ditemukan. Mohon setel variabel lingkungan YOUTUBE_API_KEY.")
        else:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("URL YouTube tidak valid. Mohon periksa kembali.")
            else:
                st.info(f"Menganalisis video: {youtube_url}")
                
                with st.spinner("Mengambil komentar dan informasi video..."):
                    comments, video_title, video_thumbnail = get_youtube_comments(video_id, youtube_api_key, max_results=500) # Limiting to 500 comments for performance

                if comments and video_title and video_thumbnail:
                    st.write("---")
                    st.subheader("Informasi Video")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(video_thumbnail, use_column_width=True)
                    with col2:
                        st.write(f"**Judul Video:** {video_title}")
                        st.write(f"**Jumlah Komentar Ditemukan:** {len(comments)}")
                        st.write(f"**Jumlah Komentar Dianalisis (Max 500):** {len(comments)}") # Display actual analyzed count


                    st.subheader("Hasil Analisis Sentimen")
                    
                    # Ensure the model is loaded
                    sentiment_classifier = None
                    try:
                        sentiment_classifier = load_sentiment_pipeline()
                    except Exception as e:
                        st.error(f"Gagal memuat model sentimen IndoBERT: {e}")
                        st.stop() # Stop execution if model can't be loaded

                    # Perform sentiment analysis
                    # Split comments into batches to avoid hitting API/model limits for large number of comments
                    # Also, the pipeline expects a list of strings
                    
                    # A small function to batch processing
                    def batch_process_comments(comment_list, batch_size=50):
                        for i in range(0, len(comment_list), batch_size):
                            yield comment_list[i:i + batch_size]

                    analyzed_data = []
                    progress_text = "Analisis sentimen sedang berjalan. Mohon tunggu..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    total_comments = len(comments)
                    processed_count = 0

                    for i, batch in enumerate(batch_process_comments(comments)):
                        try:
                            results = sentiment_classifier(batch)
                            for comment_text, result in zip(batch, results):
                                analyzed_data.append({
                                    "comment": comment_text,
                                    "sentiment": result['label'],
                                    "score": result['score']
                                })
                            processed_count += len(batch)
                            my_bar.progress(min(1.0, processed_count / total_comments), text=progress_text) # Ensure progress doesn't exceed 1.0
                        except Exception as e:
                            st.error(f"Error during sentiment analysis batch {i}: {e}")
                            # Skip this batch, or handle appropriately
                            pass
                    my_bar.empty() # Clear the progress bar after completion

                    df = pd.DataFrame(analyzed_data)

                    if not df.empty:
                        # Display sentiment distribution
                        sentiment_counts = df['sentiment'].value_counts()
                        st.write("### Distribusi Sentimen")
                        
                        # Define colors for sentiments
                        sentiment_colors = {
                            'positive': 'green',
                            'negative': 'red',
                            'neutral': 'blue'
                        }

                        fig = px.pie(
                            names=sentiment_counts.index,
                            values=sentiment_counts.values,
                            title="Distribusi Sentimen Komentar",
                            color=sentiment_counts.index,
                            color_discrete_map=sentiment_colors
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.write("### Contoh Komentar per Sentimen:")
                        for sentiment_type in ['positive', 'negative', 'neutral']:
                            st.write(f"#### {sentiment_type.capitalize()} Komentar")
                            # Filter for the sentiment type, sort by score (desc for positive, asc for negative/neutral)
                            if sentiment_type == 'positive':
                                filtered_comments = df[df['sentiment'] == sentiment_type].sort_values(by='score', ascending=False)
                            elif sentiment_type == 'negative':
                                filtered_comments = df[df['sentiment'] == sentiment_type].sort_values(by='score', ascending=False) # High score for negative means high confidence in negative
                            else: # neutral
                                filtered_comments = df[df['sentiment'] == sentiment_type].sort_values(by='score', ascending=False) # High score for neutral means high confidence in neutral
                            
                            if not filtered_comments.empty:
                                for i, row in filtered_comments.head(5).iterrows():
                                    st.markdown(f"- {row['comment']} (Score: {row['score']:.2f})")
                            else:
                                st.info(f"Tidak ada komentar {sentiment_type} ditemukan.")

                        st.write("---")
                        st.subheader("Kata Paling Dominan per Sentimen (Word Cloud)")
                        
                        for sentiment_type in ['positive', 'neutral', 'negative']:
                            col_wc_title, col_wc_chart = st.columns([1, 4])
                            with col_wc_title:
                                st.markdown(f"#### {sentiment_type.capitalize()}")
                            with col_wc_chart:
                                sentiment_comments = " ".join(df[df['sentiment'] == sentiment_type]['comment'].apply(clean_text_for_wordcloud))
                                if sentiment_comments.strip(): # Check if there's any text after cleaning
                                    generate_word_cloud(sentiment_comments, sentiment_type)
                                else:
                                    st.info(f"Tidak ada cukup data komentar {sentiment_type} untuk membuat word cloud.")

                    else:
                        st.info("Tidak ada komentar yang dapat dianalisis atau terjadi kesalahan.")
                else:
                    st.info("Tidak dapat mengambil komentar dari video ini. Mungkin tidak ada komentar, atau API key tidak valid/tidak ada akses.")