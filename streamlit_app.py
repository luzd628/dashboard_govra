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

# 1.Segmentasi
def load_model_Segmentasi():
    bundle = joblib.load('segmentasi_bundle.pkl')
    model_segmentasi = bundle["model"]
    scaler_segmentasi = bundle["scaler"]
    return model_segmentasi, scaler_segmentasi

# 2. Analisis Harga Pangan
def load_lstm():
    model_lstm = load_model('model_lstm.keras')
    scaler_lstm = joblib.load('scaler.joblib')
    params_lstm = joblib.load('lstm_params.joblib')
    return model_lstm, scaler_lstm, params_lstm

def predict_price_by_date(target_date_str):
    model_lstm, scaler_lstm, params_lstm = load_lstm()
    N_PAST = params_lstm['N_PAST']
    N_FUTURE = params_lstm['N_FUTURE']
    N_TOTAL_PREDICTION = params_lstm['N_TOTAL_PREDICTION']
    current_window = params_lstm['last_window']
    last_date = pd.to_datetime(params_lstm['last_date'])

    forecast_scaled = []
    for _ in range(0, N_TOTAL_PREDICTION, N_FUTURE):
        input_scaled = scaler_lstm.transform(current_window)
        input_batch = np.expand_dims(input_scaled, axis=0)
        prediction = model_lstm.predict(input_batch)[0]
        forecast_scaled.extend(prediction)
        prediction_real = scaler_lstm.inverse_transform(prediction)
        current_window = np.vstack([current_window, prediction_real])[-N_PAST:]

    forecast_scaled = np.array(forecast_scaled)[:N_TOTAL_PREDICTION]
    forecast_real = scaler_lstm.inverse_transform(forecast_scaled)
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(N_TOTAL_PREDICTION)]

    forecast_df = pd.DataFrame(forecast_real, columns=['Rata-rata Harga'])
    forecast_df['Tanggal'] = future_dates

    target_date = pd.to_datetime(target_date_str)
    result = forecast_df.loc[forecast_df['Tanggal'] == target_date]

    if result.empty:
        return None, forecast_df

    return float(result['Rata-rata Harga'].values[0]), forecast_df

def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained('./gpt2_beras')
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_beras')
    return model, tokenizer

def generate_policy_text(prompt):
    model, tokenizer = load_gpt2()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=500,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)



#---------------------------------------UI---------------------------------
with tab1:
    st.header("Analisis Kenaikan Harga Pangan")
    tanggal_input = st.date_input("Pilih tanggal prediksi", value=datetime(2025, 6, 15))
    if st.button("Prediksi dan Analisis"):

        harga_prediksi, df_forecast = predict_price_by_date(tanggal_input)

        if harga_prediksi is None:
            st.warning("Tanggal di luar jangkauan prediksi.")
        else:
            # Buat kalimat ringkasan prediksi
            start_price = df_forecast['Rata-rata Harga'].iloc[0]
            end_price = df_forecast['Rata-rata Harga'].iloc[-1]
            price_change_pct = ((end_price - start_price) / start_price) * 100

            df_forecast['Delta'] = df_forecast['Rata-rata Harga'].diff()
            max_delta_date = df_forecast.loc[df_forecast['Delta'].abs().idxmax(), 'Tanggal']
            max_delta_value = df_forecast['Delta'].abs().max()

            trend = "kenaikan" if end_price > start_price else "penurunan" if end_price < start_price else "stabil"

            summary = (
                f"Prediksi harga untuk {tanggal_input.strftime('%Y-%m-%d')}: Rp{harga_prediksi:,.0f}\n\n"
                f"Harga menunjukkan tren {trend} selama {len(df_forecast)} hari ke depan, "
                f"dari Rp{start_price:,.0f} menjadi Rp{end_price:,.0f} "
                f"({price_change_pct:.2f}%). Perubahan paling signifikan terjadi pada "
                f"{max_delta_date.date()} dengan selisih sekitar Rp{max_delta_value:,.0f}."
            )

            st.subheader("Hasil Prediksi")
            st.text(summary)

            # Generate dari GPT-2
            with st.spinner("Menganalisis kebijakan..."):
                gpt_output = generate_policy_text(summary)

            st.subheader("Analisis & Rekomendasi Kebijakan")
            st.text(gpt_output)

   

with tab2:
    st.header("Segmentasi Wilayah Rawan Sosial")
    with st.form("form_segmentasi"):
        wilayah = st.text_input("Nama Wilayah")
        poorpeople_percentage = st.number_input("Tingkat Kemiskinan (%)", min_value=0.0, max_value=100.0, format="%.2f")
        reg_gdp = st.number_input("Gross Domestic Product Regional (GDP) (Juta Rp)", min_value=0.0, format="%.2f") 
        life_exp = st.number_input("Angka Harapan Hidup (Tahun)", min_value=0.0, max_value=100.0, format="%.2f")
        avg_schooltime = st.number_input("Rata-rata Lama Sekolah (Tahun)", min_value=0.0, max_value=20.0, format="%.2f") 
        exp_percapita = st.number_input("Pengeluaran Per Kapita (Juta Rp)", min_value=0.0, format="%.2f") 
        submitted = st.form_submit_button("Prediksi")

        # load model & scaler
        model_segmentasi, scaler_segmentasi = load_model_Segmentasi()

        if submitted:
            data_input_segmentasi = pd.DataFrame([{
                'poorpeople_percentage': poorpeople_percentage,
                'reg_gdp': reg_gdp,
                'life_exp': life_exp,
                'avg_schooltime': avg_schooltime,
                'exp_percap': exp_percapita 
            }])

            try:
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
                st.markdown("---")
                st.subheader("Hasil Prediksi Segmentasi")
                st.success(f"Wilayah **{wilayah}** diprediksi masuk ke dalam segmen: **{predicted_label}**.")

                st.markdown("### Detail Karakteristik Input Wilayah:")
                st.write(f"- Persentase Orang Miskin: **{poorpeople_percentage:.2f}%**")
                st.write(f"- Produk Domestik Regional Bruto (PDRB): **Rp {reg_gdp:,.0f} Juta**")
                st.write(f"- Angka Harapan Hidup: **{life_exp:.2f} tahun**")
                st.write(f"- Rata-rata Lama Sekolah: **{avg_schooltime:.2f} tahun**")
                st.write(f"- Pengeluaran per Kapita: **Rp {exp_percapita:,.0f} Juta**")
            
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi segmentasi: {e}")
                st.info("Pastikan model dan scaler segmentasi (segmentasi_bundle.pkl) kompatibel dan terload dengan benar.")
            

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
        
        