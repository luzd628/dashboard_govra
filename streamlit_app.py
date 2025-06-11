import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import urllib
import altair as alt
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tensorflow.keras.preprocessing.image import img_to_array


st.title('GOVRA (Governance with AI)')

# Sidebar
with st.sidebar:
    
    st.header('Tentang Kami')
    
    st.write(" Team Capstone Laskar AI ID: LAI25-SM085 ")
    st.markdown("""
    Anggota grup
    - M Faiq Rofifi - Universitas Telkom
    - Dzul Fikri - Stmik Amikom Surakarta
    - Alifia Mustika Sari - Universitas PGRI Madiun
    - Muhammad Faizal Pratama - Universitas Teknologi Digital
    """)
    
# Main Page
st.write(' GOVRA (Governance with AI) adalah platform AI untuk membantu pemerintah kota menganalisis data sosial, ekonomi, dan lingkungan secara real-time, serta menghasilkan narasi kebijakan otomatis. Menggabungkan berbagai analisis dan LLM di Vertex AI, GOVRA mendorong tata kelola kota yang adaptif dan berbasis data.')

st.markdown("### Fitur Utama GOVRA:")
st.markdown("""
- Analisis kenaikan harga pangan
- Segmentasi wilayah rawan sosial
- Analisis sentimen publik
- Klasifikasi gambar kondisi lingkungan
""")

st.subheader('Pilih Fitur:')

tab1, tab2, tab3, tab4 = st.tabs(["Analisis Harga Pangan", "Segmentasi Wilayah Sosial", "Analisis Sentimen", "Klasifikasi Gambar Sampah"])

#----------------Function---------------------
def load_model_Segmentasi():
    bundle = joblib.load('segmentasi_bundle.pkl')
    model_segmentasi = bundle["model"]
    scaler_segmentasi = bundle["scaler"]
    return model_segmentasi, scaler_segmentasi


#---------------------------------------UI---------------------------------
with tab1:
    st.header("Analisis Kenaikan Harga Pangan")
    tanggal_input = st.date_input("Pilih tanggal prediksi", value=datetime(2025, 6, 15))
    
   

with tab2:
    st.header("Segmentasi Wilayah Rawan Sosial")
    with st.form("form_segmentasi"):
        wilayah = st.text_input("Nama Wilayah")
        poorpeople_percentage = st.number_input("Tingkat Kemiskinan (%)", min_value=0.0, max_value=100.0, format="%.2f")
        reg_gdp = st.number_input("Gross Domestic Product Regional (GDP) (Juta Rp)", min_value=0.0, format="%.2f") # Ubah max_value jika perlu
        life_exp = st.number_input("Angka Harapan Hidup (Tahun)", min_value=0.0, max_value=100.0, format="%.2f")
        avg_schooltime = st.number_input("Rata-rata Lama Sekolah (Tahun)", min_value=0.0, max_value=20.0, format="%.2f") # Menambahkan label, min/max, dan format
        exp_percapita = st.number_input("Pengeluaran Per Kapita (Juta Rp)", min_value=0.0, format="%.2f") # Menambahkan label, min_value, dan format
        submitted = st.form_submit_button("Prediksi")

        # load model & scaler
        model_segmentasi, scaler_segmentasi = load_model_Segmentasi()

        if submitted:
            data_input_segmentasi = pd.DataFrame([{
                'poorpeople_percentage': poorpeople_percentage,
                'reg_gdp': reg_gdp,
                'life_exp': life_exp,
                'avg_schooltime': avg_schooltime,
                'exp_percap': exp_percapita # Pastikan nama kolom sesuai dengan yang diharapkan model
            }])

            # Standardisasi dan prediksi
            data_input_segmentasi_scaled = scaler_segmentasi.transform(data_input_segmentasi)
            prediksi = model_segmentasi.predict(data_input_segmentasi_scaled)
            cluster_id = prediksi[0]
            # Mapping label
            cluster_labels = {
                0: "Wilayah Berkembang dengan Tingkat Kemiskinan Moderat",
                1: "Pusat Ekonomi dengan Daya Beli Tinggi"
            }
            predicted_label = cluster_labels.get(cluster_id, "Cluster tidak dikenal")

            # Output
            print(f"Wilayah ini diprediksi masuk ke dalam **{predicted_label}**.")
            print(f"Detail karakteristik wilayah:")
            print(f"- Persentase orang miskin: **{data_input_segmentasi['poorpeople_percentage'].iloc[0]}%**")
            print(f"- Produk Domestik Regional Bruto (PDRB): **Rp {data_input_segmentasi['reg_gdp'].iloc[0]:,.0f}**")
            print(f"- Angka Harapan Hidup: **{data_input_segmentasi['life_exp'].iloc[0]} tahun**")
            print(f"- Rata-rata Lama Sekolah: **{data_input_segmentasi['avg_schooltime'].iloc[0]} tahun**")
            print(f"- Pengeluaran per Kapita: **Rp {data_input_segmentasi['exp_percap'].iloc[0]:,.0f}**")
            st.write(f"Data input untuk wilayah **{wilayah}**:")

with tab3:
    st.header("Analisis Sentimen Pelayanan Publik")
    with st.form("form_sentimen"):
        teks = st.text_area("Masukkan teks opini/sentimen masyarakat")
        submitted = st.form_submit_button("Prediksi")

with tab4:
    st.header("Klasifikasi Gambar Sampah")
    with st.form("form_gambar"):
        uploaded_file = st.file_uploader("Upload gambar sampah", type=["png", "jpg", "jpeg"])
        show_code = st.sidebar.checkbox("📄 Show Backend Code")
        
        