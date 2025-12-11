# app.py — VERSI FINAL PANJANG & LENGKAP (DENGAN VISUALISASI LEBIH DETAIL & INSIGHTFUL)
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import emoji
from transformers import pipeline
import os
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import pandas as pd  # Tambah buat data processing time series
from collections import Counter  # Tambah buat top words
from datetime import datetime  # Tambah buat parse timestamp

# ================== LOAD ENV & CONFIG ==================
API_KEY = st.secrets["YOUTUBE_API_KEY"]
if not API_KEY:
    st.error("Masukkan YOUTUBE_API_KEY di secrets!")
    st.stop()

# Custom CSS — bikin cantik banget
st.markdown("""
<style>
   .stApp {
        background: linear-gradient(135deg, #0f0f23, #1a1a2e);
        color: #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #090979);
        color: white;
        border-radius: 16px;
        font-weight: bold;
        border: none;
        padding: 12px;
        box-shadow: 0 4px 15px rgba(0,212,255,0.4);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,212,255,0.6);
    }
    .stTextInput>div>div>input {
        background-color: #16213e;
        color: white;
        border-radius: 12px;
        border: 1px solid #00d4ff;
    }
    .stMarkdown, .stText {
        color: #e0e0e0 !important;
    }
    h1, h2, h3 {color: #00d4ff !important;}
    .css-1d391kg {background: transparent !important;}
</style>
""", unsafe_allow_html=True)

# Load model Indo RoBERTa (cache otomatis)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis",
                    model="w11wo/indonesian-roberta-base-sentiment-classifier",
                    truncation=True, max_length=512)

nlp = load_sentiment_model()

# ================== FUNGSI UTAMA ==================
def extract_video_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_video_info(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    try:
        req = youtube.videos().list(part="snippet", id=video_id)
        res = req.execute()
        if res['items']:
            item = res['items'][0]['snippet']
            return {'title': item['title'], 'thumbnail_url': item['thumbnails']['high']['url']}
    except HttpError as e:
        if 'quotaExceeded' in str(e):
            st.error("Quota API habis! Reset otomatis jam 15:00 WIB besok.")
        else:
            st.error("Error mengambil info video.")
    return None

def fetch_comments(video_id, max_comments=500):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    timestamps = []  # Tambah buat simpan publishedAt buat time series
    next_page_token = None
    status_text = st.empty()
    fetched = 0

    while fetched < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100, pageToken=next_page_token
            )
            response = request.execute()
            for item in response['items']:
                comment_snippet = item['snippet']['topLevelComment']['snippet']
                comments.append(comment_snippet['textDisplay'])
                timestamps.append(comment_snippet['publishedAt'])  # Tambah timestamp
                fetched += 1
                status_text.text(f"Mengambil komentar: {fetched}/{max_comments}")
                if fetched >= max_comments:
                    break
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except HttpError as e:
            if 'quotaExceeded' in str(e):
                st.error("Quota API habis! Coba lagi besok.")
            else:
                st.error("Error ambil komentar.")
            return [], []
    status_text.success(f"Berhasil ambil {len(comments)} komentar!")
    return comments, timestamps

def clean_comment(text):
    text = emoji.replace_emoji(text, "")
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def analyze_sentiment(comments, timestamps):
    cleaned = [clean_comment(c) for c in comments]
    sentiment_texts = {"positive": [], "negative": [], "neutral": []}  # Tambah buat word cloud per sentimen
    sentiment_data = []  # Tambah buat time series: label + timestamp
    
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    samples = {"positive": [], "negative": [], "neutral": []}
    
    batch_size = 64
    status = st.empty()
    status.text("Menganalisis sentimen: 0%")
    total = len(cleaned)
    for i in range(0, total, batch_size):
        batch = cleaned[i:i+batch_size]
        results = nlp(batch)
        for r, com, ts in zip(results, batch, timestamps[i:i+batch_size]):
            label = r['label'].lower()
            sentiments[label] += 1
            sentiment_texts[label].append(com)  # Kumpul teks per sentimen
            sentiment_data.append({'date': datetime.fromisoformat(ts.rstrip('Z')), 'sentimen': label})  # Time series data
            if len(samples[label]) < 5:
                samples[label].append(com)
        status.text(f"Menganalisis sentimen: {round(((i + batch_size)/total)*100)}%")
    
    percentages = {k: round(v/total*100, 2) if total else 0 for k, v in sentiments.items()}
    status.success("Analisis selesai!")
    
    return sentiments, percentages, total, samples, sentiment_texts, sentiment_data

def generate_wordcloud(text):
    if not text: return None
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    return buf

def get_top_words(texts, top_n=10):
    all_words = ' '.join(texts).split()
    counter = Counter(all_words)
    return counter.most_common(top_n)

# ================== SESSION STATE ==================
keys = ['video_info','video_id','comments','timestamps','counts','percentages','valid_comments','samples','sentiment_texts','sentiment_data']
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

# ================== UI UTAMA ==================
st.set_page_config(page_title="YT Sentiment Analyzer", layout="wide")
st.title("YouTube Sentiment Analyzer Indonesia")
st.markdown("Masukkan link → Cari video → Atur jumlah komentar → Analisis sentimen otomatis dengan insight detail!")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png")
    st.header("Pengaturan")
    slider_val = st.slider("Pilih dengan Slider", 100, 5000, 500, step=100)
    custom_val = st.number_input("Atau Ketik Manual", 100, 5000, slider_val, step=100,
                                 help="Maksimal 5000 komentar")
    max_comments = custom_val
    if max_comments > 2000:
        st.warning("Banyak komentar = proses lama & quota cepat habis!")
    show_wordcloud = st.checkbox("Tampilkan Word Cloud", True)

url = st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=...")

# Tombol Cari
if st.button("Cari Video", type="secondary") and url:
    vid = extract_video_id(url)
    if vid:
        info = fetch_video_info(vid)
        if info:
            st.session_state.video_info = info
            st.session_state.video_id = vid
            st.success(f"Ditemukan: **{info['title']}**")
            st.image(info['thumbnail_url'], width=500)
        else:
            st.error("Video tidak ditemukan.")
    else:
        st.error("Link tidak valid!")

# Tombol Mulai Analisis
if st.session_state.video_info and not st.session_state.comments:
    if st.button("Mulai Analisis Sentimen", type="primary"):
        comments, timestamps = fetch_comments(st.session_state.video_id, max_comments)
        if comments:
            c, p, v, s, texts, data = analyze_sentiment(comments, timestamps)
            st.session_state.update({'comments':comments, 'timestamps':timestamps, 'counts':c, 'percentages':p,
                                    'valid_comments':v, 'samples':s, 'sentiment_texts':texts, 'sentiment_data':data})

# Tampilkan hasil (cache tetap ada)
if st.session_state.comments:
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visualisasi Dasar", "Insight Detail", "Detail Komentar"])  # Tambah tab insight detail
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", f"{st.session_state.percentages['positive']}%", f"{st.session_state.counts['positive']} komentar")
        col2.metric("Neutral", f"{st.session_state.percentages['neutral']}%", f"{st.session_state.counts['neutral']} komentar")
        col3.metric("Negative", f"{st.session_state.percentages['negative']}%", f"{st.session_state.counts['negative']} komentar")
        st.success(f"Total komentar dianalisis: {st.session_state.valid_comments}")

    with tab2:
        fig_pie = px.pie(values=list(st.session_state.percentages.values()),
                        names=["Positif","Netral","Negatif"],
                        color_discrete_sequence=["#00ff00","#ffff00","#ff0000"])
        st.plotly_chart(fig_pie, use_container_width=True)
        fig_bar = px.bar(x=["Positif","Netral","Negatif"], y=list(st.session_state.percentages.values()))
        st.plotly_chart(fig_bar, use_container_width=True)
        if show_wordcloud:
            wc = generate_wordcloud(' '.join(st.session_state.comments))
            if wc: st.image(wc, caption="Word Cloud Semua Komentar")

    with tab3:  # Insight Detail Baru
        st.subheader("Insight Lebih Detail")
        
        # Word Cloud Per Sentimen
        if show_wordcloud:
            st.markdown("**Word Cloud Per Sentimen**")
            cols = st.columns(3)
            for idx, sent in enumerate(["positive", "negative", "neutral"]):
                text = ' '.join(st.session_state.sentiment_texts[sent])
                wc_buf = generate_wordcloud(text)
                if wc_buf:
                    cols[idx].image(wc_buf, caption=f"Word Cloud {sent.capitalize()}", use_column_width=True)
        
        # Top 10 Kata Per Sentimen
        st.markdown("**Top 10 Kata Paling Sering Per Sentimen**")
        cols = st.columns(3)
        for idx, sent in enumerate(["positive", "negative", "neutral"]):
            top_words = get_top_words(st.session_state.sentiment_texts[sent])
            if top_words:
                df_top = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])
                fig_top = px.bar(df_top, x='Frekuensi', y='Kata', orientation='h', title=f"Top Kata {sent.capitalize()}")
                cols[idx].plotly_chart(fig_top, use_container_width=True)
        
        # Sentimen Over Time (Line Chart)
        if st.session_state.sentiment_data:
            df_time = pd.DataFrame(st.session_state.sentiment_data)
            df_time['date'] = pd.to_datetime(df_time['date']).dt.date  # Group by day
            df_group = df_time.groupby(['date', 'sentimen']).size().unstack(fill_value=0)
            fig_time = px.line(df_group, x=df_group.index, y=df_group.columns, title="Sentimen Over Time")
            st.plotly_chart(fig_time, use_container_width=True)

    with tab4:
        for sent in ["positive", "neutral", "negative"]:
            with st.expander(f"{sent.capitalize()} Comments (contoh)"):
                for com in st.session_state.samples[sent]:
                    st.write(f"• {com}")

    # Download
    buf = BytesIO()
    fig_pie.write_image(buf, format="JPG")
    buf.seek(0)
    st.download_button("Download Pie Chart (JPG)", buf, "pie_chart.jpg", "image/jpeg")

    if st.button("Analisis Video Lain"):
        for k in keys:
            st.session_state[k] = None
        st.experimental_rerun()

st.caption("© 2025 — Proyek UAS Text Mining | Indo RoBERTa Powered | Dibuat bareng Grok ")

