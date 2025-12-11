# app.py
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError  # Tambah buat handle quota error
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

# ================== CONFIG ==================
API_KEY = st.secrets["YOUTUBE_API_KEY"]

# Custom CSS (sama)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f0f2f6, #ffffff);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .metric-label {
        font-size: 18px !important;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Load model (sama)
@st.cache_resource
def load_sentiment_model():
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512
    )

nlp = load_sentiment_model()

# ================== FUNGSI UTAMA ==================
def extract_video_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_video_info(video_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    try:
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if response['items']:
            item = response['items'][0]['snippet']
            return {'title': item['title'], 'thumbnail_url': item['thumbnails']['high']['url']}
    except HttpError as e:
        if 'quotaExceeded' in str(e):
            st.error("⚠️ Quota API habis! Tunggu reset jam 15:00 WIB besok atau request extension di Google Console.")
        else:
            st.error(f"Error API: {e}")
    return None

def fetch_comments(video_id, max_comments=500):  # Default 500 biar hemat
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    next_page_token = None
    
    progress_bar = st.progress(0)
    with st.spinner(f"Mengambil hingga {max_comments} komentar..."):
        total_requests = (max_comments // 100) + 1
        for i in range(total_requests):
            try:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)
                    if len(comments) >= max_comments:
                        break
                
                next_page_token = response.get('nextPageToken')
                progress_bar.progress((i + 1) / total_requests)
                if not next_page_token:
                    break
            except HttpError as e:
                if 'quotaExceeded' in str(e):
                    st.error("⚠️ Quota API habis! Tunggu reset jam 15:00 WIB besok atau request extension di Google Console.")
                else:
                    st.error(f"Error API: {e}")
                return []
    
    return comments

def clean_comment(text):
    text = emoji.replace_emoji(text, "")
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def analyze_sentiment(comments):
    cleaned = [clean_comment(c) for c in comments]
    cleaned = [c for c in cleaned if len(c) > 10]
    
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    sentiment_comments = {"positive": [], "negative": [], "neutral": []}
    
    batch_size = 64
    progress_bar = st.progress(0)
    total_batches = (len(cleaned) // batch_size) + 1
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i+batch_size]
        results = nlp(batch)
        for r, comment in zip(results, batch):
            label = r['label'].lower()
            sentiments[label] += 1
            if len(sentiment_comments[label]) < 5:
                sentiment_comments[label].append(comment)
        progress_bar.progress((i + 1) / total_batches)
        time.sleep(0.1)
    
    total = sum(sentiments.values())
    percentages = {k: round(v/total*100, 2) for k, v in sentiments.items()}
    
    return sentiments, percentages, len(cleaned), sentiment_comments

def generate_wordcloud(text, title):
    if not text:
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/youtube-play.png")
    st.title("Pengaturan")
    
    # Slider + Input Manual (SUPER USER FRIENDLY)
    st.markdown("**Jumlah Komentar yang Diambil:**")
    col1, col2 = st.columns(2)
    with col1:
        slider_val = st.slider("Geser", 100, 5000, 500, step=100, label_visibility="collapsed")
    with col2:
        manual_val = st.number_input("Ketik", 100, 5000, slider_val, step=100, label_visibility="collapsed")
    
    max_comments = manual_val  # Prioritas input manual
    
    if max_comments > 3000:
        st.error("5000 komentar = proses bisa 2-3 menit & quota langsung jebol!")
    elif max_comments > 1500:
        st.warning("Banyak komentar = lama + quota cepat habis!")
    
    show_wordcloud = st.checkbox("Tampilkan Word Cloud", value=True)
    st.markdown("---")
    st.caption("App ini pakai Indo RoBERTa untuk analisis akurat bahasa Indo. Dibuat untuk UAS Text Mining")

st.title("📊 YouTube Sentiment Analyzer (Bahasa Indonesia)")
st.markdown("Paste link YouTube → Cari dulu untuk verifikasi → Analisis sentimen komentar otomatis.")

# Inisialisasi session state (tambah buat cache hasil analisis)
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'comments' not in st.session_state:
    st.session_state.comments = None
if 'counts' not in st.session_state:
    st.session_state.counts = None
if 'percentages' not in st.session_state:
    st.session_state.percentages = None
if 'valid_comments' not in st.session_state:
    st.session_state.valid_comments = None
if 'sentiment_comments' not in st.session_state:
    st.session_state.sentiment_comments = None

url = st.text_input("🔗 Link Video YouTube", placeholder="https://www.youtube.com/watch?v=...", value="" if st.session_state.video_info else st.text_input.value)  # Kosongin kalau reset

# Tombol "Cari"
if st.button("🔍 Cari Video", type="secondary") and url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Link YouTube tidak valid!")
    else:
        with st.spinner("Mencari video..."):
            video_info = fetch_video_info(video_id)
            if video_info:
                st.session_state.video_info = video_info
                st.session_state.video_id = video_id
                st.success(f"Video ditemukan: **{video_info['title']}**")
                st.image(video_info['thumbnail_url'], caption="Thumbnail Video", use_column_width=False, width=400)
            else:
                st.error("Video tidak ditemukan atau link invalid. Coba link lain!")

# Tombol "Mulai Analisis" (muncul kalau udah cari & valid)
if st.session_state.video_info:
    if st.session_state.comments is None:  # Belum analisis, tampil tombol
        if st.button("🚀 Mulai Analisis", type="primary"):
            video_id = st.session_state.video_id
            comments = fetch_comments(video_id, max_comments=max_comments)
            if len(comments) == 0:
                st.warning("Tidak ada komentar atau komentar dinonaktifkan.")
            else:
                st.success(f"Berhasil ambil {len(comments)} komentar!")
                
                with st.spinner("Sedang menganalisis sentimen... (bisa 10-30 detik)"):
                    counts, percentages, valid_comments, sentiment_comments = analyze_sentiment(comments)
                
                # Simpan ke session_state (cache)
                st.session_state.comments = comments
                st.session_state.counts = counts
                st.session_state.percentages = percentages
                st.session_state.valid_comments = valid_comments
                st.session_state.sentiment_comments = sentiment_comments

    # Tampil hasil (dari cache kalau refresh)
    if st.session_state.comments:
        counts = st.session_state.counts
        percentages = st.session_state.percentages
        valid_comments = st.session_state.valid_comments
        sentiment_comments = st.session_state.sentiment_comments
        video_id = st.session_state.video_id
        
        tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Charts & Visual", "🔍 Details"])
        
        with tab1:
            st.subheader("Ringkasan Sentimen")
            col1, col2, col3 = st.columns(3)
            col1.metric("😊 Positive", f"{percentages['positive']}%", f"{counts['positive']} komentar")
            col2.metric("😐 Neutral", f"{percentages['neutral']}%", f"{counts['neutral']} komentar")
            col3.metric("😠 Negative", f"{percentages['negative']}%", f"{counts['negative']} komentar")
            st.markdown(f"**Total komentar dianalisis:** {valid_comments} (dari {len(st.session_state.comments)} total)")

        with tab2:
            st.subheader("Visualisasi Sentimen")
            fig_pie = px.pie(
                values=list(percentages.values()),
                names=list(percentages.keys()),
                title="Distribusi Sentimen",
                color_discrete_sequence=["#00ff00", "#ffff00", "#ff0000"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            fig_bar = px.bar(
                x=list(percentages.keys()),
                y=list(percentages.values()),
                text=list(percentages.values()),
                title="Persentase Sentimen",
                color=list(percentages.keys()),
                color_discrete_map={"positive": "#00ff00", "neutral": "#ffff00", "negative": "#ff0000"}
            )
            fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
            
            if show_wordcloud:
                st.subheader("Word Cloud Kata Kunci")
                all_text = ' '.join(sentiment_comments['positive'] + sentiment_comments['neutral'] + sentiment_comments['negative'])
                wc_buf = generate_wordcloud(all_text, "Kata Kunci Umum")
                if wc_buf:
                    st.image(wc_buf, use_column_width=True)

        with tab3:
            st.subheader("Sample Komentar")
            with st.expander("😊 Top Positive Comments"):
                for c in sentiment_comments['positive']:
                    st.write(f"- {c}")
            with st.expander("😐 Top Neutral Comments"):
                for c in sentiment_comments['neutral']:
                    st.write(f"- {c}")
            with st.expander("😠 Top Negative Comments"):
                for c in sentiment_comments['negative']:
                    st.write(f"- {c}")
        
        # Download PDF & JPG
        def create_pdf():
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            
            c.setFont("Helvetica-Bold", 20)
            c.drawString(50, height-80, "Laporan Analisis Sentimen YouTube")
            c.setFont("Helvetica", 12)
            c.drawString(50, height-110, f"Video ID: {video_id}")
            c.drawString(50, height-130, f"Total komentar dianalisis: {valid_comments}")
            
            y = height - 180
            for sentiment, pct in percentages.items():
                c.drawString(50, y, f"{sentiment.capitalize()}: {pct}% ({counts[sentiment]} komentar)")
                y -= 25
            
            pie_img = BytesIO()
            fig_pie.write_image(pie_img, format="PNG")
            pie_img.seek(0)
            c.drawImage(ImageReader(pie_img), 50, y-300, width=500, height=300)
            
            c.showPage()
            c.save()
            buffer.seek(0)
            return buffer
        
        pdf_buffer = create_pdf()
        st.download_button(
            "📥 Download Laporan PDF",
            pdf_buffer,
            file_name=f"sentimen_{video_id}.pdf",
            mime="application/pdf",
            key="pdf_download"
        )
        
        jpg_buffer = BytesIO()
        fig_pie.write_image(jpg_buffer, format="JPG")
        jpg_buffer.seek(0)
        st.download_button(
            "🖼️ Download Pie Chart (JPG)",
            jpg_buffer,
            file_name=f"pie_chart_{video_id}.jpg",
            mime="image/jpeg",
            key="jpg_download"
        )
        
        # Tombol "Analisis Video Lain" (reset cache)
        if st.button("🔄 Analisis Video Lain", type="secondary"):
            # Reset session_state
            st.session_state.video_info = None
            st.session_state.video_id = None
            st.session_state.comments = None
            st.session_state.counts = None
            st.session_state.percentages = None
            st.session_state.valid_comments = None
            st.session_state.sentiment_comments = None
            st.experimental_rerun()  # Refresh app biar input kosong

st.caption("Dibuat dengan ❤️ dan Indo RoBERTa — cocok banget buat tugas Text Mining | Versi Optimasi Quota & Cache")