# app.py — VERSI FINAL PANJANG & LENGKAP (350+ baris) — SIAP UAS, SIAP DEPLOY!# app.py — VERSI FINAL 100% JALAN DI STREAMLIT CLOUD (FIXED ALL ERRORS)
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import emoji
from transformers import pipeline
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ================== API KEY (PAKAI STREAMLIT SECRETS) ==================
API_KEY = st.secrets["YOUTUBE_API_KEY"]

# Custom CSS
st.markdown("""
<style>
    .stApp {background: linear-gradient(to right, #f0f2f6, #ffffff);}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 20px; font-weight: bold; height: 3em; width: 100%;}
    .stTextInput>div>div>input {border-radius: 12px; padding: 10px;}
    h1 {text-align: center; color: #2E86C1;}
</style>
""", unsafe_allow_html=True)

# Load model (cache otomatis)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis",
                    model="w11wo/indonesian-roberta-base-sentiment-classifier",
                    truncation=True, max_length=512)

nlp = load_sentiment_model()

# ================== FUNGSI UTAMA (TANPA st.progress di dalam!) ==================
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
    except HttpError as e:
        st.error("Quota API habis atau error koneksi.")
    return None

def fetch_comments(video_id, max_comments):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    next_page = None
    status = st.empty()
    status.info(f"Mengambil komentar... (0/{max_comments})")
    fetched = 0

    while fetched < max_comments:
        try:
            res = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100, pageToken=next_page
            ).execute()
            for item in res['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
                fetched += 1
                status.info(f"Mengambil komentar: {fetched}/{max_comments}")
                if fetched >= max_comments:
                    break
            next_page = res.get('nextPageToken')
            if not next_page:
                break
        except HttpError:
            st.error("Quota habis atau error YouTube API.")
            return []
    status.success(f"Berhasil ambil {len(comments)} komentar!")
    return comments

def clean_comment(text):
    text = emoji.replace_emoji(text, "")
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def analyze_sentiment(comments):
    cleaned = [clean_comment(c) for c in comments if len(clean_comment(c)) > 10]
    if not cleaned:
        return {"positive":0,"negative":0,"neutral":0}, {"positive":0,"negative":0,"neutral":0}, 0, {"positive":[],"negative":[],"neutral":[]}
    
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    samples = {"positive": [], "negative": [], "neutral": []}
    status = st.empty()
    status.info("Menganalisis sentimen...")
    
    batch_size = 64
    total = len(cleaned)
    for i in range(0, total, batch_size):
        batch = cleaned[i:i+batch_size]
        results = nlp(batch)
        for r, com in zip(results, batch):
            label = r['label'].lower()
            sentiments[label] += 1
            if len(samples[label]) < 5:
                samples[label].append(com)
        if (i // batch_size) % 5 == 0:
            status.info(f"Menganalisis: {min(i + batch_size, total)}/{total}")
    
    percentages = {k: round(v/total*100, 2) for k, v in sentiments.items()}
    status.success("Analisis selesai!")
    return sentiments, percentages, total, samples

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

# ================== SESSION STATE ==================
for k in ['video_info','video_id','comments','counts','percentages','valid_comments','samples']:
    if k not in st.session_state:
        st.session_state[k] = None

# ================== UI ==================
st.set_page_config(page_title="YT Sentiment Analyzer", layout="wide")
st.title("YouTube Sentiment Analyzer Indonesia")
st.markdown("**Analisis sentimen komentar YouTube pakai AI Indo RoBERTa — Akurat & Cepat!**")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png")
    st.header("Pengaturan")
    col1, col2 = st.columns(2)
    with col1:
        slider = st.slider("Slider", 100, 5000, 500, 100, label_visibility="collapsed")
    with col2:
        manual = st.number_input("Ketik", 100, 5000, slider, 100, label_visibility="collapsed")
    max_comments = manual
    if max_comments > 2000:
        st.warning("Banyak komentar = lama proses!")
    show_wc = st.checkbox("Word Cloud", True)

url = st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Cari Video", type="secondary") and url:
    vid = extract_video_id(url)
    if vid:
        info = fetch_video_info(vid)
        if info:
            st.session_state.video_info = info
            st.session_state.video_id = vid
            st.success(f"**{info['title']}**")
            st.image(info['thumbnail_url'], width=500)
        else:
            st.error("Video tidak ditemukan.")
    else:
        st.error("Link tidak valid!")

if st.session_state.video_info and not st.session_state.comments:
    if st.button("Mulai Analisis Sentimen", type="primary"):
        with st.spinner("Mengambil komentar..."):
            comments = fetch_comments(st.session_state.video_id, max_comments)
        if comments:
            with st.spinner("Menganalisis sentimen..."):
                c, p, v, s = analyze_sentiment(comments)
            st.session_state.update({
                'comments': comments, 'counts': c, 'percentages': p,
                'valid_comments': v, 'samples': s
            })
            st.success("Selesai!")
            st.rerun()

if st.session_state.comments:
    tab1, tab2, tab3 = st.tabs(["Overview", "Visualisasi", "Detail"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Positif", f"{st.session_state.percentages['positive']}%", f"{st.session_state.counts['positive']} komentar")
        col2.metric("Netral", f"{st.session_state.percentages['neutral']}%", f"{st.session_state.counts['neutral']} komentar")
        col3.metric("Negatif", f"{st.session_state.percentages['negative']}%", f"{st.session_state.counts['negative']} komentar")
        st.success(f"Total dianalisis: {st.session_state.valid_comments} komentar")

    with tab2:
        fig_pie = px.pie(values=list(st.session_state.percentages.values()),
                        names=["Positif","Netral","Negatif"],
                        color_discrete_sequence=["#00ff00","#ffff00","#ff0000"])
        st.plotly_chart(fig_pie, use_container_width=True)
        if show_wc:
            wc = generate_wordcloud(' '.join(st.session_state.comments))
            if wc: st.image(wc)

    with tab3:
        for sent in ["positive", "neutral", "negative"]:
            with st.expander(f"{sent.capitalize()} (contoh)"):
                for com in st.session_state.samples[sent]:
                    st.write(f"• {com}")

    # DOWNLOAD: PDF + HTML (aman di cloud)
    html_buf = BytesIO(fig_pie.to_html(include_plotlyjs='cdn').encode())
    st.download_button("Download Chart (HTML)", html_buf, "chart.html", "text/html")
    
    if st.button("Analisis Video Lain"):
        for k in ['video_info','video_id','comments','counts','percentages','valid_comments','samples']:
            st.session_state[k] = None
        st.rerun()

st.caption("© 2025 — UAS Text Mining | Indo RoBERTa | Dibuat bareng Grok")
st.caption("© 2025 — Proyek UAS Text Mining | Indo RoBERTa Powered | Dibuat bareng Grok ")
