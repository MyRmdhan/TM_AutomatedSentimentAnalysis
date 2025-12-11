# app.py — VERSI FINAL DARK MODE + VISUALISASI LEBIH VARIATIF & INFORMATIF + SEMUA FITUR KAMU MINTA
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import emoji
from transformers import pipeline
import plotly.express as px
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer  # Tambah buat TF-IDF kata berpengaruh

# ================== API KEY ==================
API_KEY = st.secrets["YOUTUBE_API_KEY"]

# FULL DARK MODE — CANTIK, MODERN, ELEGAN
st.set_page_config(page_title="YT Sentiment Analyzer", layout="wide")
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a, #1a1a1a);
        color: #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ffffff, #888888);
        color: #000000;
        border-radius: 16px;
        font-weight: bold;
        border: none;
        padding: 12px;
        box-shadow: 0 4px 15px rgba(255,255,255,0.2);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255,255,255,0.3);
    }
    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: white;
        border-radius: 12px;
        border: 1px solid #333333;
    }
    h1, h2, h3, h4 {color: #ffffff !important;}
    .stMarkdown {color: #e0e0e0 !important;}
    .stExpander {background-color: #1a1a1a; border: 1px solid #333333; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis",
                    model="w11wo/indonesian-roberta-base-sentiment-classifier",
                    truncation=True, max_length=512)

nlp = load_sentiment_model()

# ================== FUNGSI ==================
def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None

def fetch_video_info(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    try:
        res = youtube.videos().list(part="snippet", id=video_id).execute()
        if res['items']:
            item = res['items'][0]['snippet']
            return {'title': item['title'], 'thumbnail_url': item['thumbnails']['high']['url']}
    except HttpError:
        st.error("Quota habis atau error API.")
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
            res = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100, pageToken=next_page
            ).execute()
            for item in res['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append(snippet['textDisplay'])
                timestamps.append(snippet['publishedAt'])
                fetched += 1
                if fetched % 50 == 0:
                    status.info(f"Mengambil: {fetched}/{max_comments}")
                if fetched >= max_comments: break
            next_page = res.get('nextPageToken')
            if not next_page: break
        except HttpError:
            st.error("Quota habis!")
            return [], []
    status.success(f"Berhasil ambil {len(comments)} komentar!")
    return comments, timestamps

def clean_comment(text):
    text = emoji.replace_emoji(text, "")
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def analyze_sentiment(comments, timestamps):
    cleaned = []
    valid_timestamps = []
    for c, ts in zip(comments, timestamps):
        cleaned_text = clean_comment(c)
        if len(cleaned_text) > 10:
            cleaned.append(cleaned_text)
            valid_timestamps.append(ts)

    if not cleaned:
        return {}, {}, 0, {}, {}, [], []

    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    samples = {"positive": [], "negative": [], "neutral": []}
    texts = {"positive": [], "negative": [], "neutral": []}
    data = []
    scores = {"positive": [], "negative": [], "neutral": []}  # Tambah buat distribution scores

    status = st.empty()
    status.info("Menganalisis sentimen...")
    total = len(cleaned)
    batch_size = 64

    for i in range(0, total, batch_size):
        batch = cleaned[i:i+batch_size]
        batch_ts = valid_timestamps[i:i+batch_size]
        results = nlp(batch)
        for r, com, ts in zip(results, batch, batch_ts):
            label = r['label'].lower()
            sentiments[label] += 1
            texts[label].append(com)
            scores[label].append(r['score'])  # Tambah score buat box plot
            if len(samples[label]) < 5:
                samples[label].append(com)
            data.append({'date': datetime.fromisoformat(ts.replace('Z','+00:00')), 'sentimen': label})
        if (i // batch_size) % 3 == 0:
            status.info(f"Proses: {min(i + batch_size, total)}/{total}")

    percentages = {k: round(v/total*100, 2) for k, v in sentiments.items()}
    status.success("Analisis selesai!")

    # Tambah TF-IDF buat kata berpengaruh
    tfidf = TfidfVectorizer(max_features=20)
    tfidf_docs = [ ' '.join(texts[sent]) for sent in ['positive', 'negative', 'neutral'] ]
    tfidf.fit(tfidf_docs)
    tfidf_words = {}
    for i, sent in enumerate(['positive', 'negative', 'neutral']):
        feature_names = tfidf.get_feature_names_out()
        scores_vec = tfidf.transform([tfidf_docs[i]]).toarray()[0]
        tfidf_words[sent] = sorted([(feature_names[j], scores_vec[j]) for j in range(len(scores_vec))], key=lambda x: x[1], reverse=True)

    return sentiments, percentages, total, samples, texts, data, scores, tfidf_words

def generate_wordcloud(text):
    if not text: return None
    wc = WordCloud(width=800, height=400, background_color='#0f0f23', colormap='viridis').generate(text)
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#0f0f23')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0f0f23', dpi=150)
    buf.seek(0); plt.close()
    return buf

# ================== SESSION STATE ==================
for k in ['video_info','video_id','comments','timestamps','counts','percentages','valid_comments','samples','sentiment_texts','sentiment_data','scores','tfidf_words']:
    if k not in st.session_state:
        st.session_state[k] = None

# ================== UI ==================
st.title("YouTube Sentiment Analyzer")
st.markdown("**Dark Mode • Indo RoBERTa • Insight Lebih Variatif & Detail • Siap Presentasi UAS**")

with st.sidebar:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0a0a, #1a1a1a);
        }
    </style>
    """, unsafe_allow_html=True)
    
    col_img = st.columns([1])[0]
    col_img.image("https://img.icons8.com/color/96/youtube-play.png", width=50)
    
    st.markdown("---")
    st.markdown("### ⚙️ Pengaturan Analisis")
    st.markdown("Atur parameter untuk menganalisis komentar video YouTube Anda")
    
    st.markdown("#### 📊 Jumlah Komentar")
    st.markdown("Pilih jumlah komentar yang ingin dianalisis (100 - 5000)")
    col1, col2 = st.columns(2)
    with col1: 
        slider = st.slider("Slider", 100, 5000, 500, 100, label_visibility="collapsed")
    with col2: 
        manual = st.number_input("Manual", 100, 5000, slider, 100, label_visibility="collapsed")
    max_comments = manual
    
    if max_comments > 2000: 
        st.warning("⏱️ Jumlah komentar banyak, proses akan memakan waktu lebih lama")
    elif max_comments > 1000:
        st.info("✓ Jumlah komentar optimal untuk analisis")
    
    st.markdown("---")
    st.markdown("#### 🎨 Visualisasi")
    show_wc = st.checkbox("☁️ Tampilkan Word Cloud", value=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #333333, #555555); padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;'>
        <p style='color: white; margin: 0; font-size: 12px;'><strong>💡 Tips:</strong> Gunakan 500-1000 komentar untuk hasil optimal dan proses cepat</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888; font-size: 11px;'>v1.0 • Dark Mode • Indo RoBERTa</p>", unsafe_allow_html=True)

url = st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=...")

if not st.session_state.comments:
    if st.button("Cari Video", type="secondary") and url:
        vid = extract_video_id(url)
        if vid:
            info = fetch_video_info(vid)
            if info:
                st.session_state.video_info = info
                st.session_state.video_id = vid
                st.success(f"**{info['title']}**")
                st.image(info['thumbnail_url'], width=500)
            else: st.error("Video tidak ditemukan.")
        else: st.error("Link tidak valid!")

    if st.session_state.video_info and not st.session_state.comments:
        if st.button("Mulai Analisis Sentimen", type="primary"):
            with st.spinner("Proses penuh sedang berjalan..."):
                comments, timestamps = fetch_comments(st.session_state.video_id, max_comments)
                if comments:
                    result = analyze_sentiment(comments, timestamps)
                    c, p, v, s, texts, data, scores, tfidf_words = result
                    st.session_state.update({
                        'comments': comments, 'timestamps': timestamps, 'counts': c, 'percentages': p,
                        'valid_comments': v, 'samples': s, 'sentiment_texts': texts, 'sentiment_data': data, 'scores': scores, 'tfidf_words': tfidf_words
                    })
                    st.success("Selesai!"); st.rerun()

if st.session_state.comments:
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Charts Dasar", "Insight Detail", "Sample Komentar"])

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
        if show_wc:
            st.subheader("Word Cloud Per Sentimen")
            cols = st.columns(3)
            for i, sent in enumerate(["positive", "negative", "neutral"]):
                wc = generate_wordcloud(' '.join(st.session_state.sentiment_texts[sent]))
                if wc: cols[i].image(wc, caption=sent.capitalize())

        st.subheader("Top Kata Paling Banyak Per Sentimen")
        cols = st.columns(3)
        for i, sent in enumerate(["positive", "negative", "neutral"]):
            top = Counter(' '.join(st.session_state.sentiment_texts[sent]).split()).most_common(10)
            if top:
                df = pd.DataFrame(top, columns=['Kata', 'Frekuensi'])
                fig_top = px.bar(df, x='Frekuensi', y='Kata', orientation='h', template="plotly_dark", title=sent.capitalize())
                cols[i].plotly_chart(fig_top, use_container_width=True)

        st.subheader("Kata Berpengaruh (TF-IDF) Per Sentimen")
        cols = st.columns(3)
        for i, sent in enumerate(["positive", "negative", "neutral"]):
            top_tfidf = st.session_state.tfidf_words[sent]
            if top_tfidf:
                df_tfidf = pd.DataFrame(top_tfidf, columns=['Kata', 'Score'])
                fig_tfidf = px.bar(df_tfidf, x='Score', y='Kata', orientation='h', template="plotly_dark", title=sent.capitalize())
                cols[i].plotly_chart(fig_tfidf, use_container_width=True)

        st.subheader("Distribusi Confidence Score Per Sentimen")
        score_df = pd.DataFrame()
        for sent in ["positive", "negative", "neutral"]:
            if st.session_state.scores[sent]:
                temp_df = pd.DataFrame({'Sentimen': sent, 'Score': st.session_state.scores[sent]})
                score_df = pd.concat([score_df, temp_df])
        fig_box = px.box(score_df, x='Sentimen', y='Score', template="plotly_dark", title="Distribusi Score")
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Tren Sentimen Over Time")
        if st.session_state.sentiment_data:
            df = pd.DataFrame(st.session_state.sentiment_data)
            df['date'] = pd.to_datetime(df['date']).dt.floor('H')
            dfg = df.groupby([pd.Grouper(key='date', freq='6H'), 'sentimen']).size().unstack(fill_value=0)
            fig = px.line(dfg, template="plotly_dark", title="Tren per 6 Jam")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        for sent in ["positive", "neutral", "negative"]:
            with st.expander(f"{sent.capitalize()} (contoh)"):
                for c in st.session_state.samples[sent]:
                    st.write(f"• {c}")

    st.divider()
    if st.button("Analisis Video Lain", type="primary"):
        keys = ['video_info','video_id','comments','timestamps','counts','percentages','valid_comments','samples','sentiment_texts','sentiment_data','scores','tfidf_words']
        for k in keys: st.session_state[k] = None
        st.rerun()

st.caption("© 2025 — Dark Mode Edition | Indo RoBERTa | Dibuat bareng Grok")
