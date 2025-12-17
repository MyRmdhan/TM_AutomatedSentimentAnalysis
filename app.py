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
st.set_page_config(page_title="YT Sentiment Analyzer", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a, #1a1a1a);
        color: #e0e0e0;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #ffffff, #cccccc);
        color: #000000;
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
        box-shadow: 0 4px 15px rgba(255,255,255,0.15);
        transition: all 0.3s ease;
        font-size: 14px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255,255,255,0.25);
    }
    
    /* Input Styling */
    .stTextInput>div>div>input {
        background-color: #151515;
        color: #ffffff;
        border-radius: 10px;
        border: 1.5px solid #333333;
        padding: 12px 16px;
        font-size: 14px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #666666;
        box-shadow: 0 0 0 3px rgba(255,255,255,0.1);
    }
    
    /* Slider Styling */
    .stSlider>div>div>div>div {
        background-color: #333333;
    }
    
    /* Heading Styling */
    h1 {
        color: #ffffff !important;
        font-size: 2.5em !important;
        font-weight: 700 !important;
        margin-bottom: 10px !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #ffffff !important;
        font-size: 1.8em !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
    }
    
    h3, h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    /* Expander Styling */
    .stExpander {
        background-color: #151515;
        border: 1px solid #333333;
        border-radius: 10px;
    }
    
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
    }
    
    /* Metric Styling */
    .stMetric {
        background-color: #151515;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        color: #999999;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #333333;
        color: #ffffff;
    }
    
    /* Success/Info/Warning Messages */
    .stSuccess {
        background-color: rgba(0,200,0,0.1) !important;
        border-left: 4px solid #00c800 !important;
    }
    
    .stInfo {
        background-color: rgba(100,200,255,0.1) !important;
        border-left: 4px solid #64c8ff !important;
    }
    
    .stWarning {
        background-color: rgba(255,200,0,0.1) !important;
        border-left: 4px solid #ffc800 !important;
    }
    
    /* Divider */
    .stDivider {
        margin: 20px 0 !important;
        border: none;
        border-top: 1px solid #333333 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a, #151515);
        border-right: 1px solid #333333;
    }
    
    /* Selectbox & Number Input */
    .stNumberInput>div>div>input {
        background-color: #151515;
        color: #ffffff;
        border-radius: 10px;
        border: 1.5px solid #333333;
        padding: 10px 12px;
    }
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
# Indonesian stopwords untuk visualisasi saja
INDO_STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'adalah', 'pada', 'atau', 'tidak',
    'ini', 'itu', 'dengan', 'oleh', 'akan', 'telah', 'sudah', 'dapat', 'juga',
    'lebih', 'pula', 'dalam', 'ada', 'karena', 'bagian', 'anda', 'saya', 'dia',
    'mereka', 'kami', 'kalian', 'aku', 'kamu', 'dia', 'ia', 'nya',
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'it', 'its', 'of', 'in', 'on', 'at',
    'to', 'as', 'by', 'or', 'but', 'if', 'because', 'so'
}

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
    # Hapus HTML tags dan entities
    text = re.sub(r'<[^>]+>', '', text)  # Hapus HTML tags
    text = re.sub(r'&nbsp;|&lt;|&gt;|&amp;|&quot;|&#39;|&.*?;', '', text)  # Hapus HTML entities
    
    # Hapus emoji
    text = emoji.replace_emoji(text, "")
    
    # Hapus URL
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Hapus newline dan carriage return
    text = re.sub(r'[\n\r\t]+', ' ', text)
    
    # Hapus multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip whitespace di awal dan akhir, lowercase
    text = text.strip().lower()
    
    # Hapus special characters tapi jaga huruf, angka, dan spasi
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    return text

def clean_for_visualization(text):
    """Hapus stopwords dari teks untuk kepentingan visualisasi saja"""
    words = text.split()
    filtered = [w for w in words if w not in INDO_STOPWORDS and len(w) > 2]
    return ' '.join(filtered)

def analyze_sentiment(comments, timestamps):
    cleaned = []
    valid_timestamps = []
    for c, ts in zip(comments, timestamps):
        cleaned_text = clean_comment(c)
        if len(cleaned_text) > 10:
            cleaned.append(cleaned_text)
            valid_timestamps.append(ts)

    if not cleaned:
        return {}, {}, 0, {}, {}, {}, [], [], {}

    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    samples = {"positive": [], "negative": [], "neutral": []}
    texts_original = {"positive": [], "negative": [], "neutral": []}  # Teks asli untuk IndoBERT
    texts_clean = {"positive": [], "negative": [], "neutral": []}  # Teks bersih untuk visualisasi
    data = []
    scores = {"positive": [], "negative": [], "neutral": []}

    status = st.empty()
    status.info("Menganalisis sentimen...")
    total = len(cleaned)
    batch_size = 64

    # Step 1: Kirim teks asli ke IndoBERT untuk analisis
    for i in range(0, total, batch_size):
        batch = cleaned[i:i+batch_size]
        batch_ts = valid_timestamps[i:i+batch_size]
        results = nlp(batch)
        for r, com, ts in zip(results, batch, batch_ts):
            label = r['label'].lower()
            sentiments[label] += 1
            texts_original[label].append(com)  # Simpan teks asli
            
            # Bersihkan untuk visualisasi
            clean_text = clean_for_visualization(com)
            if clean_text:  # Hanya simpan jika ada kata setelah pembersihan
                texts_clean[label].append(clean_text)
            
            scores[label].append(r['score'])
            if len(samples[label]) < 5:
                samples[label].append(com)
            data.append({'date': datetime.fromisoformat(ts.replace('Z','+00:00')), 'sentimen': label})
        if (i // batch_size) % 3 == 0:
            status.info(f"Proses: {min(i + batch_size, total)}/{total}")

    percentages = {k: round(v/total*100, 2) for k, v in sentiments.items()}
    status.success("Analisis selesai!")

    # Step 2: Hitung TF-IDF menggunakan teks bersih untuk visualisasi
    tfidf = TfidfVectorizer(max_features=20)
    tfidf_docs = [' '.join(texts_clean[sent]) for sent in ['positive', 'negative', 'neutral']]
    # Cek apakah ada dokumen kosong
    if all(doc.strip() for doc in tfidf_docs):
        tfidf.fit(tfidf_docs)
        tfidf_words = {}
        for i, sent in enumerate(['positive', 'negative', 'neutral']):
            feature_names = tfidf.get_feature_names_out()
            scores_vec = tfidf.transform([tfidf_docs[i]]).toarray()[0]
            tfidf_words[sent] = sorted([(feature_names[j], scores_vec[j]) for j in range(len(scores_vec))], key=lambda x: x[1], reverse=True)
    else:
        tfidf_words = {"positive": [], "negative": [], "neutral": []}

    return sentiments, percentages, total, samples, texts_clean, data, scores, tfidf_words, texts_original

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
for k in ['video_info','video_id','comments','timestamps','counts','percentages','valid_comments','samples','sentiment_texts','sentiment_data','scores','tfidf_words','sentiment_texts_original','comments_raw','timestamps_raw','scraped','is_running']:
    if k not in st.session_state:
        st.session_state[k] = None

# ================== UI ==================
# Header Section dengan styling menarik
col_header1, col_header2 = st.columns([1, 1])
with col_header1:
    st.markdown("""
    <div style='padding: 20px 0;'>
        <h1 style='margin: 0;'>🎬 YouTube Sentiment Analyzer</h1>
        <p style='color: #888888; font-size: 14px; margin: 8px 0 0 0;'>Analisis sentimen komentar video YouTube dengan AI IndoBERT</p>
    </div>
    """, unsafe_allow_html=True)

with col_header2:
    # Simple, unobtrusive header right area for version only
    st.markdown("""
    <div style='text-align: right; padding: 20px 0;'>
        <p style='color: #666666; font-size: 12px; margin:0;'>v1.0</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border: none; border-top: 1px solid #333333; margin: 20px 0;'>", unsafe_allow_html=True)

# Sidebar removed — move configuration controls into the main page after the video is found

if 'show_wc' not in st.session_state:
    st.session_state['show_wc'] = True
if 'max_comments' not in st.session_state:
    st.session_state['max_comments'] = 500

url = st.text_input("Link YouTube", placeholder="https://www.youtube.com/watch?v=...", help="Masukkan URL video YouTube yang ingin dianalisis")

if not st.session_state.comments:
    col_btn1, col_btn2, col_placeholder = st.columns([2, 3, 2])
    
    with col_btn1:
        if st.button("🔍 Cari Video", type="secondary", use_container_width=True) and url:
            vid = extract_video_id(url)
            if vid:
                info = fetch_video_info(vid)
                if info:
                    st.session_state.video_info = info
                    st.session_state.video_id = vid
                else:
                    st.markdown("""
                    <div style='background: rgba(200,0,0,0.15); border-left: 4px solid #c80000; padding: 12px; border-radius: 10px;'>
                        <p style='color: #ff6666; font-size: 12px; margin: 0;'><strong>✗ Video tidak ditemukan</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            else: 
                st.markdown("""
                <div style='background: rgba(200,0,0,0.15); border-left: 4px solid #c80000; padding: 12px; border-radius: 10px;'>
                    <p style='color: #ff6666; font-size: 12px; margin: 0;'><strong>✗ Link YouTube tidak valid</strong></p>
                </div>
                """, unsafe_allow_html=True)

    if st.session_state.video_info and not st.session_state.comments:
        # Persistent preview: show title + thumbnail so it doesn't disappear on reruns
        with col_btn2:
            info = st.session_state.get('video_info')
            if info:
                st.markdown(f"""
                <div style='background: rgba(0,200,100,0.15); border-left: 4px solid #00c864; padding: 12px; border-radius: 10px; margin: 8px 0;'>
                    <p style='color: #66ff99; font-size: 13px; font-weight: 600; margin: 0;'>✓ Video Ditemukan</p>
                    <p style='color: #cccccc; font-size: 12px; margin: 6px 0 6px 0;'><strong>{info['title']}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.image(info['thumbnail_url'], use_container_width=True)

        
            # Initialize widget-backed session_state keys (avoid setting during render)
            if 'slider_max_comments' not in st.session_state:
                st.session_state['slider_max_comments'] = st.session_state.get('max_comments', 500)
            if 'manual_max_comments' not in st.session_state:
                st.session_state['manual_max_comments'] = st.session_state.get('max_comments', 500)

            # Keep slider and manual input synchronized via session_state callbacks
            def _sync_slider():
                # when slider changes, update manual and canonical max_comments
                val = st.session_state.get('slider_max_comments')
                st.session_state['max_comments'] = val
                st.session_state['manual_max_comments'] = val

            def _sync_manual():
                # when manual input changes, update slider and canonical max_comments
                val = st.session_state.get('manual_max_comments')
                st.session_state['max_comments'] = val
                st.session_state['slider_max_comments'] = val

            c1, c2 = st.columns(2)
            with c1:
                slider = st.slider("Jumlah Komentar (100-5000)", 100, 5000, key='slider_max_comments', step=100, on_change=_sync_slider)
            with c2:
                manual = st.number_input("Manual", 100, 5000, key='manual_max_comments', step=100, on_change=_sync_manual)

            if st.session_state['max_comments'] > 2000:
                st.markdown("""
                <div style='background: rgba(255,150,0,0.15); border-left: 3px solid #ff9600; padding: 10px; border-radius: 6px; margin: 10px 0;'>
                    <p style='color: #ffb366; font-size: 12px; margin: 0;'><strong>⏱️ Catatan:</strong> Banyak komentar = proses lebih lama</p>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state['max_comments'] >= 1000:
                st.markdown("""
                <div style='background: rgba(0,200,100,0.15); border-left: 3px solid #00c864; padding: 10px; border-radius: 6px; margin: 10px 0;'>
                    <p style='color: #66ff99; font-size: 12px; margin: 0;'><strong>✓ Optimal:</strong> Parameter sudah ideal untuk analisis cepat</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.session_state['show_wc'] = st.checkbox("☁️ Tampilkan Word Cloud", value=st.session_state.get('show_wc', True))

          

            # Action button aligned full-width
            if st.button("▶️ Mulai Analisis Sentimen", type="primary", use_container_width=True) and not st.session_state.get('is_running'):
                st.session_state['is_running'] = True
                try:
                    # Step 1: Scrape comments only (store raw)
                    with st.spinner("Mengambil komentar dari YouTube..."):
                        comments_raw, timestamps_raw = fetch_comments(st.session_state.video_id, st.session_state['max_comments'])
                        if not comments_raw:
                            st.error("Gagal mengambil komentar atau tidak ada komentar.")
                            st.session_state['is_running'] = False
                        else:
                            st.session_state['comments_raw'] = comments_raw
                            st.session_state['timestamps_raw'] = timestamps_raw
                            st.session_state['scraped'] = True

                    # Step 2: Run analysis on scraped comments
                    if st.session_state.get('scraped'):
                        with st.spinner("Menganalisis komentar..."):
                            result = analyze_sentiment(st.session_state['comments_raw'], st.session_state['timestamps_raw'])
                            c, p, v, s, texts, data, scores, tfidf_words, texts_original = result
                            st.session_state.update({
                                'comments': st.session_state['comments_raw'], 'timestamps': st.session_state['timestamps_raw'],
                                'counts': c, 'percentages': p,
                                'valid_comments': v, 'samples': s, 'sentiment_texts': texts, 'sentiment_data': data, 'scores': scores, 'tfidf_words': tfidf_words, 'sentiment_texts_original': texts_original
                            })
                            st.success("✓ Analisis selesai!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat proses: {e}")
                    st.session_state['is_running'] = False

if st.session_state.comments:
    st.markdown("<hr style='border: none; border-top: 1px solid #333333; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-bottom: 30px;'>📊 Hasil Analisis Sentimen</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📉 Statistik", "🔍 Insight Detail", "💬 Sample"])

    with tab1:
        # Metrics dengan styling modern
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1a3a1a, #2a5a2a); padding: 20px; border-radius: 12px; border: 1px solid #3a7a3a; text-align: center;'>
                <p style='color: #88dd88; font-size: 12px; margin: 0 0 8px 0;'>POSITIF</p>
                <p style='color: #00ff00; font-size: 32px; font-weight: 700; margin: 0;'>""" + str(st.session_state.percentages['positive']) + """%</p>
                <p style='color: #66cc66; font-size: 12px; margin: 8px 0 0 0;'>""" + str(st.session_state.counts['positive']) + """ komentar</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #3a3a1a, #5a5a2a); padding: 20px; border-radius: 12px; border: 1px solid #7a7a3a; text-align: center;'>
                <p style='color: #dddd88; font-size: 12px; margin: 0 0 8px 0;'>NETRAL</p>
                <p style='color: #ffff00; font-size: 32px; font-weight: 700; margin: 0;'>""" + str(st.session_state.percentages['neutral']) + """%</p>
                <p style='color: #cccc66; font-size: 12px; margin: 8px 0 0 0;'>""" + str(st.session_state.counts['neutral']) + """ komentar</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #3a1a1a, #5a2a2a); padding: 20px; border-radius: 12px; border: 1px solid #7a3a3a; text-align: center;'>
                <p style='color: #dd8888; font-size: 12px; margin: 0 0 8px 0;'>NEGATIF</p>
                <p style='color: #ff4488; font-size: 32px; font-weight: 700; margin: 0;'>""" + str(st.session_state.percentages['negative']) + """%</p>
                <p style='color: #cc6666; font-size: 12px; margin: 8px 0 0 0;'>""" + str(st.session_state.counts['negative']) + """ komentar</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: linear-gradient(135deg, #2a2a2a, #3a3a3a); padding: 15px; border-radius: 10px; border: 1px solid #444444;'>
            <p style='color: #cccccc; font-size: 13px; margin: 0;'><strong>📝 Total Komentar Dianalisis:</strong> """ + str(st.session_state.valid_comments) + """ komentar</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<p style='color: #999999; font-size: 13px; margin-bottom: 20px;'>Pie chart distribusi sentimen dari seluruh komentar yang dianalisis</p>", unsafe_allow_html=True)
        fig = px.pie(values=list(st.session_state.percentages.values()),
                     names=["Positif","Netral","Negatif"],
                     color_discrete_sequence=["#00ff88","#ffff88","#ff4488"],
                     template="plotly_dark",
                     hole=0.3)
        fig.update_layout(height=450, font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if st.session_state.get('show_wc', True):
            st.markdown("<h3 style='margin-top: 0;'>☁️ Word Cloud Per Sentimen</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #999999; font-size: 12px; margin-top: -15px;'>Visualisasi kata-kata yang paling sering muncul (sudah dibersihkan dari stopwords)</p>", unsafe_allow_html=True)
            cols = st.columns(3, gap="medium")
            for i, sent in enumerate(["positive", "negative", "neutral"]):
                wc = generate_wordcloud(' '.join(st.session_state.sentiment_texts[sent]))
                if wc:
                    with cols[i]:
                        st.image(wc, caption=f"Sentimen: {sent.upper()}", use_container_width=True)
            st.markdown("<hr style='border: none; border-top: 1px solid #333333; margin: 30px 0;'>", unsafe_allow_html=True)

        st.markdown("<h3>🏆 Top 10 Kata Paling Sering Muncul Per Sentimen</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #999999; font-size: 12px; margin-top: -15px;'>Frekuensi kemunculan kata dalam komentar (tanpa stopwords)</p>", unsafe_allow_html=True)
        cols = st.columns(3, gap="medium")
        for i, sent in enumerate(["positive", "negative", "neutral"]):
            with cols[i]:
                top = Counter(' '.join(st.session_state.sentiment_texts[sent]).split()).most_common(10)
                if top:
                    df = pd.DataFrame(top, columns=['Kata', 'Frekuensi'])
                    fig_top = px.bar(df, x='Frekuensi', y='Kata', orientation='h', 
                                    template="plotly_dark", title=f"{sent.upper()}",
                                    color='Frekuensi', color_continuous_scale='Viridis')
                    fig_top.update_layout(height=400, showlegend=False, font=dict(size=11))
                    st.plotly_chart(fig_top, use_container_width=True)
        
        st.markdown("<hr style='border: none; border-top: 1px solid #333333; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3>⭐ Kata Berpengaruh (TF-IDF) Per Sentimen</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #999999; font-size: 12px; margin-top: -15px;'>Kata yang paling signifikan untuk setiap kategori sentimen</p>", unsafe_allow_html=True)
        cols = st.columns(3, gap="medium")
        for i, sent in enumerate(["positive", "negative", "neutral"]):
            with cols[i]:
                top_tfidf = st.session_state.tfidf_words[sent]
                if top_tfidf:
                    df_tfidf = pd.DataFrame(top_tfidf, columns=['Kata', 'Score'])
                    fig_tfidf = px.bar(df_tfidf, x='Score', y='Kata', orientation='h', 
                                      template="plotly_dark", title=f"{sent.upper()}",
                                      color='Score', color_continuous_scale='Turbo')
                    fig_tfidf.update_layout(height=400, showlegend=False, font=dict(size=11))
                    st.plotly_chart(fig_tfidf, use_container_width=True)
        
        st.markdown("<hr style='border: none; border-top: 1px solid #333333; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3>📊 Distribusi Confidence Score Per Sentimen</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #999999; font-size: 12px; margin-top: -15px;'>Tingkat kepercayaan prediksi model IndoBERT</p>", unsafe_allow_html=True)
        score_df = pd.DataFrame()
        for sent in ["positive", "negative", "neutral"]:
            if st.session_state.scores[sent]:
                temp_df = pd.DataFrame({'Sentimen': sent, 'Score': st.session_state.scores[sent]})
                score_df = pd.concat([score_df, temp_df])
        fig_box = px.box(score_df, x='Sentimen', y='Score', template="plotly_dark", 
                        title="Box Plot Distribusi Score", color='Sentimen',
                        color_discrete_map={'positive': '#00ff88', 'neutral': '#ffff88', 'negative': '#ff4488'})
        fig_box.update_layout(height=450, font=dict(size=12))
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("<hr style='border: none; border-top: 1px solid #333333; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3>📈 Tren Sentimen Over Time</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #999999; font-size: 12px; margin-top: -15px;'>Perubahan sentimen komentar setiap 6 jam</p>", unsafe_allow_html=True)
        if st.session_state.sentiment_data:
            df = pd.DataFrame(st.session_state.sentiment_data)
            df['date'] = pd.to_datetime(df['date']).dt.floor('H')
            dfg = df.groupby([pd.Grouper(key='date', freq='6H'), 'sentimen']).size().unstack(fill_value=0)
            fig = px.line(dfg, template="plotly_dark", title="Tren Sentimen per 6 Jam",
                         markers=True, line_shape='spline')
            fig.update_layout(height=450, font=dict(size=12), hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("<p style='color: #999999; font-size: 12px; margin-bottom: 20px;'>Contoh komentar dari setiap kategori sentimen</p>", unsafe_allow_html=True)
        for sent in ["positive", "neutral", "negative"]:
            with st.expander(f"💬 {sent.upper()} - {len(st.session_state.samples[sent])} Contoh", expanded=False):
                if st.session_state.samples[sent]:
                    for idx, c in enumerate(st.session_state.samples[sent], 1):
                        st.markdown(f"""
                        <div style='background: #1a1a1a; padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 3px solid {'#00ff88' if sent == 'positive' else '#ffff88' if sent == 'neutral' else '#ff4488'};'>
                            <p style='color: #cccccc; font-size: 12px; margin: 0;'><strong>{idx}.</strong> {c}</p>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: none; border-top: 1px solid #333333; margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
    
    col_btn_reset1, col_btn_reset2, col_btn_reset3 = st.columns([2, 2, 3])
    with col_btn_reset1:
        if st.button("🔄 Analisis Video Lain", type="primary", use_container_width=True):
            keys = ['video_info','video_id','comments','timestamps','counts','percentages','valid_comments','samples','sentiment_texts','sentiment_data','scores','tfidf_words','sentiment_texts_original']
            for k in keys: st.session_state[k] = None
            st.rerun()

st.markdown("<hr style='border: none; border-top: 1px solid #333333; margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666666; font-size: 11px; margin: 0;'>© 2025 • YouTube Sentiment Analyzer • Dark Mode Modern Edition v1.0</p>", unsafe_allow_html=True)

