import datetime
import random
import requests
import streamlit as st

st.title('GOVRA (Governance with AI)')

# Sidebar
with st.sidebar:
    
    st.header('Tentang Kami')
    
    values = st.slider(
        label='Select a range of values',
        min_value=0, max_value=100, value=(0, 100)
    )
    st.write('Values:', values)

# Main Page
st.write(' GOVRA (Governance with AI) adalah platform AI untuk membantu pemerintah kota menganalisis data sosial, ekonomi, dan lingkungan secara real-time, serta menghasilkan narasi kebijakan otomatis. Menggabungkan berbagai analisis dan LLM di Vertex AI, GOVRA mendorong tata kelola kota yang adaptif dan berbasis data.')

st.subheader('Pilihan Fitur:')

tab1, tab2, tab3, tab4 = st.tabs(["Tab 1", "Tab 2", "Tab 3", "Tab 4"])
 
# Fungsi Prediksi
def prediksi_harga_pangan(data):
    # url = "https://alamat-api-kamu/predict-harga-pangan"
    # try:
    #     response = requests.post(url, json=data)
    #     response.raise_for_status()
    #     hasil = response.json()
    #     return hasil.get("prediksi", "Tidak ada hasil prediksi")
    # except Exception as e:
    #     return f"Error saat memganggil API: {e}"
    return f"Prediksi dampak kenaikan harga {data['komoditas']} di wilayah {data['wilayah']}: Tinggi"

def prediksi_segmentasi(data):
    # url = "https://alamat-api-kamu/predict-segmentasi"
    # try:
    #     response = requests.post(url, json=data)
    #     response.raise_for_status()
    #     hasil = response.json()
    #     return hasil.get("segmentasi", "Tidak ada hasil segmentasi")
    # except Exception as e:
    #     return f"Error saat memanggil API: {e}"
    return f"Wilayah {data['wilayah']} termasuk kategori: Rawan Sosial Tinggi"

def prediksi_sentimen(teks):
    # url = "https://alamat-api-kamu/predict-sentimen"
    # try:
    #     response = requests.post(url, json=data)
    #     response.raise_for_status()
    #     hasil = response.json()
    #     return hasil.get("sentimen", "Tidak ada hasil sentimen")
    # except Exception as e:
    #     return f"Error saat memanggil API: {e}"
    return random.choice(["Positif", "Negatif", "Netral"])

def prediksi_gambar(file):
    # url = "https://alamat-api-kamu/predict-klasifikasi-gambar"
    # try:
    #     files = {"file": (file.name, file.getvalue(), file.type)}
    #     response = requests.post(url, files=files)
    #     response.raise_for_status()
    #     hasil = response.json()
    #     return hasil.get("kelas", "Tidak ada hasil klasifikasi")
    # except Exception as e:
    #     return f"Error saat memanggil API: {e}"
    return "Sampah Plastik"

with tab1:
    st.header("Analisis Kenaikan Harga Pangan")
    with st.form("form_harga_pangan"):
        komoditas = st.selectbox("Nama Komoditas", ["Beras", "Cabai", "Telur", "Minyak Goreng", "Gula", "Daging Ayam"])
        wilayah = st.text_input("Wilayah")
        harga_sekarang = st.number_input("Harga Saat Ini (Rp/kg)", min_value=0.0, format="%.2f")
        persentase_kenaikan = st.number_input("Persentase Kenaikan (%)", format="%.2f")
        tanggal = st.date_input("Tanggal Pengamatan", value=datetime.datetime.today().date())
        waktu = st.time_input("Waktu Pengamatan", value=datetime.datetime.now().time())
        keterangan = st.text_area("Keterangan Tambahan (opsional)")
        submitted = st.form_submit_button("Prediksi")

        if submitted:
            datetime_pengamatan = datetime.datetime.combine(tanggal, waktu)
            data = {
                "komoditas": komoditas,
                "wilayah": wilayah,
                "harga_sekarang": harga_sekarang,
                "persentase_kenaikan": persentase_kenaikan,
                "datetime_pengamatan": datetime_pengamatan,
                "keterangan": keterangan,
            }
            hasil = prediksi_harga_pangan(data)
            st.success("Prediksi selesai!")
            st.write(hasil)

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
            hasil = prediksi_segmentasi(data)
            st.success("Prediksi selesai!")
            st.write(hasil)

with tab3:
    st.header("Analisis Sentimen Publik")
    with st.form("form_sentimen"):
        teks = st.text_area("Masukkan teks opini/sentimen masyarakat")
        submitted = st.form_submit_button("Prediksi")

        if submitted:
            hasil = prediksi_sentimen(teks)
            st.success("Prediksi selesai!")
            st.write(f"Hasil Sentimen: **{hasil}**")

with tab4:
    st.header("Klasifikasi Gambar Sampah")
    with st.form("form_gambar"):
        uploaded_file = st.file_uploader("Upload gambar sampah", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Prediksi")

        if submitted:
            if uploaded_file is not None:
                hasil = prediksi_gambar(uploaded_file)
                st.success("Prediksi selesai!")
                st.write(f"Jenis Sampah: **{hasil}**")
            else:
                st.error("Silakan upload gambar terlebih dahulu!")