# app.py — FINAL: DARK MODE + STOPWORD KHUSUS VISUALISASI + 100% JALAN!
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import emoji
from transformers import pipeline
import plotly.express as px
from io import BytesIO  # INI YANG LUPA DIIMPORT! SUDAH DITAMBAH
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from collections import Counter

# ================== STOPWORDS INDONESIA ==================
INDONESIAN_STOPWORDS = {
    'yang', 'di', 'ke', 'dari', 'dan', 'dengan', 'untuk', 'pada', 'adalah', 'ini', 'itu',
    'saya', 'kamu', 'dia', 'kami', 'kita', 'mereka', 'aku', 'gue', 'lo', 'lu', 'gw',
    'banget', 'bgt', 'nih', 'si', 'ga', 'gak', 'nggak', 'kalo', 'kalau', 'bisa', 'bikin',
    'aja', 'sih', 'dong', 'deh', 'ya', 'lah', 'kok', 'kan', 'lo', 'lu', 'gw', 'ku',
    'dulu', 'baru', 'udah', 'masih', 'lagi', 'sama', 'juga', 'atau', 'karena', 'soalnya',
    'makanya', 'biar', 'supaya', 'buat', 'mau', 'pengen', 'suka', 'lebih', 'kurang'
}

# ================== API KEY ==================
API_KEY = st.secrets["YOUTUBE_API_KEY"]

# DARK MODE CANTIK
st.set_page_config(page_title="YT Sentiment Analyzer", layout="wide")
st.markdown("""
<style>
    .stApp {background: linear-gradient(135deg, #0f0f23, #1a1a2e); color: #e0e0e0;}
    .stButton>button {background: linear-gradient(45deg, #00d4ff, #090979); color: white; border-radius: 16px; font-weight: bold; border: none; padding: 12px; box-shadow: 0 4px 15px rgba(0,212,255,0.4);}
    .stButton>button:hover {transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,212,255,0.6);}
    .stTextInput>div>div>input {background-color: #16213e; color: white; border-radius: 12px; border: 1px solid #00d4ff;}
    h1, h2, h3, h4 {color: #00d4ff !important;}
    .stExpander {background-color: #16213e; border: 1px solid #00d4ff; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis",
                    model="w11wo/indonesian-roberta-base-sentiment-classifier",
                    truncation=True, max_length=512)

nlp = load_model()

# ================== PREPROCESSING ==================
def clean_for_model(text):
    text = emoji.replace_emoji(text, "")
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def clean_for_visualization(text):
    text = clean_for_model(text)
    words = text.split()
    words = [w for w in words if w not in INDONESIAN_STOPWORDS and len(w) > 2]
    return ' '.join(words)

# ================== FUNGSI LAIN ==================
def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

def fetch_video_info(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    try:
        res = youtube.videos().list(part="snippet", id=video_id).execute()
        if res['items']:
            item = res['items'][0]['snippet']
            thumb = item['thumbnails']
            url = (thumb.get('maxres', {}).get('url') or
                   thumb.get('high', {}).get('url') or
                   thumb.get('medium', {}).get('url') or
                   thumb.get('default', {}).get('url') or
                   f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
            return {'title': item['title'], 'thumbnail_url': url}
    except: pass
    return None

def fetch_comments(video_id, max_comments):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments, timestamps = [], []
    next_page = None
    status = st.empty()
    status.info("Mengambil komentar...")
    fetched = 0

    while fetched < max_comments:
        try:
            res = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, pageToken=next_page).execute()
            for item in res['items']:
                s = item['snippet']['topLevelComment']['snippet']
                comments.append(s['textDisplay'])
                timestamps.append(s['publishedAt'])
                fetched += 1
                if fetched % 50 == 0: status.info(f"Mengambil: {fetched}/{max_comments}")
                if fetched >= max_comments: break
            next_page = res.get('nextPageToken')
            if not next_page: break
        except HttpError: st.error("Quota habis!"); return [], []
    status.success(f"Berhasil ambil {len(comments)} komentar!")
    return comments, timestamps

def analyze_sentiment(comments, timestamps):
    model_texts = [clean_for_model(c) for c in comments]
    valid_indices = [i for i, t in enumerate(model_texts) if len(t) > 10]
    model_texts = [model_texts[i] for i in valid_indices]
    valid_timestamps = [timestamps[i] for i in valid_indices]

    if not model_texts: return {}, {}, 0, {}, {}, []

    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    samples = {"positive": [], "negative": [], "neutral": []}
    clean_texts = {"positive": [], "negative": [], "neutral": []}
    data = []

    status = st.empty()
    status.info("Menganalisis sentimen...")
    for i in range(0, len(model_texts), 64):
        batch = model_texts[i:i+64]
        results = nlp(batch)
        for r, idx in zip(results, valid_indices[i:i+64]):
            label = r['label'].lower()
            original_comment = comments[idx]
            sentiments[label] += 1
            clean_vis = clean_for_visualization(original_comment)
            clean_texts[label].append(clean_vis)
            if len(samples[label]) < 5: samples[label].append(original_comment[:200] + "...")
            data.append({'date': datetime.fromisoformat(timestamps[idx].replace('Z','+00:00')), 'sentimen': label})

    percentages = {k: round(v/len(model_texts)*100, 2) for k, v in sentiments.items()}
    status.success("Analisis selesai!")
    return sentiments, percentages, len(model_texts), samples, clean_texts, data

def generate_wordcloud(text):
    if not text or len(text.strip()) == 0:
        return None
    wc = WordCloud(width=800, height=400, background_color='#0f0f23', colormap='viridis', max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#0f0f23')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f0f23', dpi=150)
    buf.seek(0)
    plt.close()
    return buf

# ================== SESSION STATE ==================
for k in ['video_info','video_id','comments','timestamps','counts','percentages','valid_comments','samples','clean_texts','sentiment_data']:
    if k not in st.session_state: st.session_state[k] = None

# ================== UI ==================
st.title("YouTube Sentiment Analyzer")
st.markdown("**Dark Mode • IndoBERT Pakai Teks Asli • Visualisasi Super Bersih (Stopword Dihapus)**")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png")
    st.header("Pengaturan")
    col1, col2 = st.columns(2)
    with col1: slider = st.slider("Slider", 100, 5000, 500, 100, label_visibility="collapsed")
    with col2: manual = st.number_input("Ketik", 100, 5000, slider, 100, label_visibility="collapsed")
    max_comments = manual
    if max_comments > 2000: st.warning("Banyak komentar = lama proses!")

url = st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Cari Video", type="secondary") and url:
    vid = extract_video_id(url)
    if vid:
        info = fetch_video_info(vid)
        if info:
            st.session_state.video_info = info
            st.session_state.video_id = vid
            st.success(f"**{info['title']}**")
            st.image(info['thumbnail_url'], width=600)
        else: st.error("Video tidak ditemukan.")
    else: st.error("Link tidak valid!")

if st.session_state.video_info and not st.session_state.comments:
    if st.button("Mulai Analisis Sentimen", type="primary"):
        with st.spinner("Sedang mengambil & menganalisis..."):
            comments, timestamps = fetch_comments(st.session_state.video_id, max_comments)
            if comments:
                c, p, v, s, clean_texts, data = analyze_sentiment(comments, timestamps)
                st.session_state.update({
                    'comments': comments, 'timestamps': timestamps, 'counts': c, 'percentages': p,
                    'valid_comments': v, 'samples': s, 'clean_texts': clean_texts, 'sentiment_data': data
                })
                st.success("Selesai!"); st.rerun()

if st.session_state.comments:
    tab1, tab2, tab3 = st.tabs(["Overview", "Charts", "Insight Kata"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Positif", f"{st.session_state.percentages['positive']}%", f"{st.session_state.counts['positive']} komentar")
        col2.metric("Netral", f"{st.session_state.percentages['neutral']}%", f"{st.session_state.counts['neutral']} komentar")
        col3.metric("Negatif", f"{st.session_state.percentages['negative']}%", f"{st.session_state.counts['negative']} komentar")
        st.success(f"Total dianalisis: {st.session_state.valid_comments} komentar")

    with tab2:
        fig = px.pie(values=list(st.session_state.percentages.values()),
                     names=["Positif","Netral","Negatif"],
                     color_discrete_sequence=["#00ff88","#ffff88","#ff4488"],
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Word Cloud (Teks Bersih — Tanpa Stopword)")
        cols = st.columns(3)
        for i, sent in enumerate(["positive", "negative", "neutral"]):
            text = ' '.join(st.session_state.clean_texts[sent])
            wc = generate_wordcloud(text)
            if wc: cols[i].image(wc, caption=sent.capitalize())

        st.subheader("Top 10 Kata Paling Sering (Setelah Stopword Dihapus)")
        cols = st.columns(3)
        for i, sent in enumerate(["positive", "negative", "neutral"]):
            words = ' '.join(st.session_state.clean_texts[sent]).split()
            top = Counter(words).most_common(10)
            if top:
                df = pd.DataFrame(top, columns=['Kata', 'Frekuensi'])
                fig = px.bar(df, x='Frekuensi', y='Kata', orientation='h', template="plotly_dark", title=sent.capitalize())
                cols[i].plotly_chart(fig, use_container_width=True)

    if st.button("Analisis Video Lain"):
        for k in ['video_info','video_id','comments','timestamps','counts','percentages','valid_comments','samples','clean_texts','sentiment_data']:
            st.session_state[k] = None
        st.rerun()

st.caption("© 2025 — Final Version | IndoBERT Akurat | Visualisasi Bersih | Dibuat bareng Grok")
