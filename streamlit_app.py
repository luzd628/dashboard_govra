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
 
@st.cache_resource(allow_output_mutation=True)
def load_artifacts():
    model = load_model('model_sentimen.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def prediksi_sentimen(teks):
    seq = tokenizer.texts_to_sequences([teks])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    pred = model.predict(padded)[0]
    label_index = np.argmax(pred)
    label_map = {0: 'negatif', 1: 'netral', 2: 'positif'}
    label = label_map[label_index]
    confidence = pred[label_index]
    return label, confidence, pred  

#------------ Analisis Kenaikan Harga Pangan----------
@st.cache_resource
def load_lstm():
    model = load_model('model_lstm.keras')
    scaler = joblib.load('scaler.joblib')
    params = joblib.load('lstm_params.joblib')
    return model, scaler, params

def predict_price_by_date(target_date_str):
    model, scaler, params = load_lstm()
    N_PAST = params['N_PAST']
    N_FUTURE = params['N_FUTURE']
    N_TOTAL_PREDICTION = params['N_TOTAL_PREDICTION']
    current_window = params['last_window']
    last_date = pd.to_datetime(params['last_date'])

    forecast_scaled = []
    for _ in range(0, N_TOTAL_PREDICTION, N_FUTURE):
        input_scaled = scaler.transform(current_window)
        input_batch = np.expand_dims(input_scaled, axis=0)
        prediction = model.predict(input_batch)[0]
        forecast_scaled.extend(prediction)
        prediction_real = scaler.inverse_transform(prediction)
        current_window = np.vstack([current_window, prediction_real])[-N_PAST:]

    forecast_scaled = np.array(forecast_scaled)[:N_TOTAL_PREDICTION]
    forecast_real = scaler.inverse_transform(forecast_scaled)
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(N_TOTAL_PREDICTION)]

    forecast_df = pd.DataFrame(forecast_real, columns=['Rata-rata Harga'])
    forecast_df['Tanggal'] = future_dates

    target_date = pd.to_datetime(target_date_str)
    result = forecast_df.loc[forecast_df['Tanggal'] == target_date]

    if result.empty:
        return None, forecast_df

    return float(result['Rata-rata Harga'].values[0]), forecast_df


# --- Fungsi: Generate teks dengan GPT-2 ---
@st.cache_resource
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

#--------------------Segmentasi Wilayah Sosial---------------


#--------------------Klasifikasi Gambar Sampah-------------- 
# ------------------ Load Model ------------------
@st.cache_resource
def load_custom_model():
    return load_model("model.h5")

model = load_custom_model()

# ------------------ Load Labels ------------------
@st.cache_data
def load_labels(path="labels.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

CLASS_NAMES = load_labels()

# ------------------ Validasi Konsistensi ------------------
try:
    output_neurons = model.output_shape[-1]
    if len(CLASS_NAMES) != output_neurons:
        st.error(f"❌ Jumlah label ({len(CLASS_NAMES)}) tidak sesuai dengan output model ({output_neurons}).\n\nPeriksa kembali `labels.txt` dan arsitektur model.")
        st.stop()
except Exception as e:
    st.error(f"❌ Gagal memvalidasi label dan output model: {e}")
    st.stop()

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
        tingkat_kemiskinan = st.number_input("Tingkat Kemiskinan (%)", min_value=0.0, max_value=100.0, format="%.2f")
        tingkat_pengangguran = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=100.0, format="%.2f")
        tingkat_kejahatan = st.number_input("Tingkat Kejahatan (%)", min_value=0.0, max_value=100.0, format="%.2f")
        submitted = st.form_submit_button("Prediksi")

        if submitted:
            data = {
                "wilayah": wilayah,
                "kemiskinan": tingkat_kemiskinan,
                "pengangguran": tingkat_pengangguran,
                "kejahatan": tingkat_kejahatan,
            }
            #hasil = prediksi_segmentasi(data)
            st.success("Prediksi selesai!")
            s#t.write(hasil)

with tab3:
    st.header("Analisis Sentimen Pelayanan Publik")
    with st.form("form_sentimen"):
        teks = st.text_area("Masukkan teks opini/sentimen masyarakat")
        submitted = st.form_submit_button("Prediksi")

        model, tokenizer = load_artifacts()
        max_length = 200

        input_teks = st.text_area("📝 Masukkan opini publik atau keluhan:")

        if st.button("🔍 Analisis"):
            if input_teks.strip():
                label, confidence, scores = prediksi_sentimen(input_teks)
                label_map_display = {"negatif": "tidak puas", "netral": "netral", "positif": "puas"}

                st.markdown(f"""
                ### 📢 Hasil Analisis

                Model memprediksi bahwa kalimat:
        "{input_teks.strip()}"
                termasuk dalam sentimen {label.upper()} dengan tingkat kepercayaan sebesar {confidence*100:.1f}%.
                Hal ini menunjukkan bahwa pengguna kemungkinan merasa {label_map_display[label]} terhadap isi kalimat tersebut.
                """)

                st.markdown("---")
                st.subheader("📊 Skor Probabilitas Tiap Kelas:")
                st.write({
                    "Negatif": f"{scores[0]*100:.2f}%",
                    "Netral": f"{scores[1]*100:.2f}%",
                    "Positif": f"{scores[2]*100:.2f}%"
                })

                st.bar_chart({
                    "Negatif": scores[0],
                    "Netral": scores[1],
                    "Positif": scores[2]
                })
            else:
                st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")

with tab4:
    st.header("Klasifikasi Gambar Sampah")
    with st.form("form_gambar"):
        uploaded_file = st.file_uploader("Upload gambar sampah", type=["png", "jpg", "jpeg"])
        show_code = st.sidebar.checkbox("📄 Show Backend Code")

        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")

                # Preprocessing
                input_shape = (150, 150)
                resized_image = image.resize(input_shape)
                image_array = img_to_array(resized_image)
                image_array = np.expand_dims(image_array, axis=0) / 255.0

                # Predict
                prediction = model.predict(image_array)[0]
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Layout dua kolom
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### 🖼️ Uploaded Image")
                    st.image(image, use_container_width=True)
                with col2:
                    st.markdown("### 🧠 Prediction Result")
                    st.success(f"**Category**: {predicted_class}  \n**Confidence**: {confidence * 100:.2f}%")

                    df_result = pd.DataFrame({
                        "Class": CLASS_NAMES,
                        "Confidence": prediction
                    }).sort_values(by="Confidence", ascending=True)
                    
                    chart = alt.Chart(df_result).mark_bar().encode(
                        x=alt.X("Confidence:Q", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("Class:N", sort="-x"),
                        color=alt.value("#0E79B2")
                    ).properties(
                        width="container",
                        height=300,
                        title="Confidence per Class"
                    )
                    st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ Error saat memproses gambar: {e}")
        else:
            st.info("Silakan unggah gambar melalui sidebar untuk memulai klasifikasi.")

        # ------------------ Tampilkan Kode (Opsional) ------------------
        if show_code:
            @st.cache_data(show_spinner=False)
            def get_file_content_as_string(path):
                url = "https://raw.githubusercontent.com/alouvre/capstone_imageclassification_trashwaste/main/" + path
                response = urllib.request.urlopen(url)
                return response.read().decode("utf-8")

            st.markdown("### 📄 Backend Code")
            st.code(get_file_content_as_string("ml_frontend.py"))