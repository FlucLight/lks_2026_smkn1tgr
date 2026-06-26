import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi UMKM LKS", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('model_umkm1')

model = load_model()

feature_names = [
    'Age', 'Education', 'Initial_Capital', 'Financial_Record_Keeping',
    'Internet_Usage', 'Business_Plan', 'Marketing_Effort', 'Partnership',
    'Parent_Business_Experience', 'Industry_Experience',
    'Owner_Gender', 'Professional_Advice'   
]

st.title("Prediksi Peluang Keberhasilan UMKM")
st.write(
    "Isi profil usaha Anda di bawah ini untuk mengetahui peluang keberhasilan "
    "usaha berdasarkan analisis kecerdasan artifisial."
)
st.divider()

st.subheader("Profil Pelaku Usaha")

col_kiri, col_kanan = st.columns(2)

with col_kiri:
    age = st.slider("Usia Pemilik (tahun)", min_value=18, max_value=65, value=30)

    education_label = st.selectbox(
        "Lulusan Terakhir",
        options=["SD", "SMP", "SMA / SMK", "D3 / Diploma", "S1 ke atas"]
    )
    education_map = {"SD": 1, "SMP": 2, "SMA / SMK": 3, "D3 / Diploma": 4, "S1 ke atas": 5}
    education = education_map[education_label]

    initial_capital_label = st.radio(
        "Apakah memiliki modal awal yang cukup?",
        options=["Ya", "Tidak"],
        horizontal=True
    )
    initial_capital = 1 if initial_capital_label == "Ya" else 0

    financial_label = st.radio(
        "Rutin melakukan pencatatan keuangan?",
        options=["Ya", "Tidak"],
        horizontal=True
    )
    financial = 1 if financial_label == "Ya" else 0

    internet_label = st.radio(
        "Aktif menggunakan internet untuk bisnis?",
        options=["Ya", "Tidak"],
        horizontal=True
    )
    internet = 1 if internet_label == "Ya" else 0

    business_plan_label = st.radio(
        "Memiliki rencana usaha (business plan)?",
        options=["Ya", "Tidak"],
        horizontal=True
    )
    business_plan = 1 if business_plan_label == "Ya" else 0

with col_kanan:
    marketing = st.slider(
        "Intensitas Upaya Pemasaran",
        min_value=1, max_value=7, value=4,
        help="1 = Tidak melakukan pemasaran, 2 = hampir tidak melakukan pemasaran, 3 = sedikit, 4 = kadang, 5 = sering, 6 = sangat sering 7 = selalu melakukan"
    )

    partnership_label = st.radio(
        "Memiliki mitra atau rekanan usaha?",
        options=["Ya", "Tidak"],
        horizontal=True
    )
    partnership = 1 if partnership_label == "Ya" else 0

    parent_exp_label = st.radio(
        "Orang tua memiliki pengalaman berbisnis?",
        options=["Ya", "Tidak"],
        horizontal=True
    )
    parent_exp = 1 if parent_exp_label == "Ya" else 0

    industry_exp = st.slider(
        "Pengalaman di industri terkait (tahun)",
        min_value=0, max_value=30, value=0
    )

    gender_label = st.radio(
        "Jenis Kelamin Pemilik",
        options=["Laki-laki", "Perempuan"],
        horizontal=True
    )
    gender = 1 if gender_label == "Laki-laki" else 0

    professional_advice = st.slider(
        "Akses ke Mentor / Konsultan Bisnis",
        min_value=1, max_value=7, value=1,
        help="1 = Tidak pernah, 2 = Hampir tidak pernah, 3 = jarang, 4 = Kadang Kadang, 5 = Sering, 6 = sangat sering  7 = Selalu"
    )

st.divider()

prediksi_btn = st.button(
    "Prediksi Peluang Keberhasilan",
    type="primary",
    use_container_width=True
)

if prediksi_btn:
    input_data = pd.DataFrame([[
        age, education, initial_capital, financial,
        internet, business_plan, marketing, partnership,
        parent_exp, industry_exp, gender, professional_advice
    ]], columns=feature_names)

    proba = model.predict_proba(input_data)[0][1]
    label = model.predict(input_data)[0]
    persentase = round(proba * 100, 2)

    st.divider()
    st.subheader("Hasil Prediksi")

    col_hasil1, col_hasil2 = st.columns(2)

    with col_hasil1:
        st.metric(label="Peluang Keberhasilan", value=f"{persentase}%")

    with col_hasil2:
        if label == 1:
            st.success("Peluang Tinggi — Profil usaha Anda menunjukkan potensi keberhasilan yang baik.")
        else:
            st.warning("Peluang Rendah — Terdapat beberapa faktor yang perlu diperkuat.")

    st.divider()

    st.subheader("Faktor yang Paling Berpengaruh")

    try:
        rf_step = model.named_steps.get('model', None)
        if rf_step is not None and hasattr(rf_step, 'feature_importances_'):
            importances = rf_step.feature_importances_
        else:
            importances = model.feature_importances_

        importance_df = pd.DataFrame({
            'Faktor': feature_names,
            'Pengaruh': importances
        }).sort_values('Pengaruh', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df['Faktor'], importance_df['Pengaruh'], color='steelblue')
        ax.set_xlabel("Tingkat Pengaruh")
        ax.set_title("Feature Importance")
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        st.pyplot(fig)

    except Exception as e:
        st.info("Grafik feature importance tidak tersedia untuk model ini.")

    st.divider()

    st.subheader("Saran untuk Meningkatkan Peluang Keberhasilan")

    saran = []

    if business_plan == 0:
        saran.append("Buatlah rencana usaha (business plan) yang jelas sebagai panduan arah bisnis Anda.")
    if financial == 0:
        saran.append("Mulailah melakukan pencatatan keuangan secara rutin untuk memudahkan pengambilan keputusan bisnis.")
    if internet == 0:
        saran.append("Manfaatkan internet dan media sosial untuk memperluas jangkauan pemasaran usaha Anda.")
    if marketing <= 2:
        saran.append("Tingkatkan upaya pemasaran, baik secara offline maupun melalui platform digital dan marketplace.")
    if partnership == 0:
        saran.append("Pertimbangkan membangun kemitraan untuk membuka akses ke pasar dan sumber daya yang lebih luas.")
    if professional_advice <= 2:
        saran.append("Cari akses ke mentor atau konsultan bisnis. Dinas Koperasi dan UMKM Provinsi Kalimantan Timur menyediakan program pendampingan usaha.")
    if initial_capital == 0:
        saran.append("Eksplorasi sumber permodalan seperti KUR (Kredit Usaha Rakyat) atau program bantuan modal dari pemerintah daerah.")

    if saran:
        for item in saran:
            st.info(item)
    else:
        st.success("Profil usaha Anda sudah sangat baik di semua faktor utama. Pertahankan dan terus kembangkan usaha Anda.")

    st.divider()
    st.caption(
        "Hasil prediksi ini bersifat sebagai alat bantu pengambilan keputusan. "
        "Keputusan akhir tetap berada di tangan pelaku usaha dan sebaiknya "
        "dikonsultasikan dengan pihak yang berkompeten."
    )