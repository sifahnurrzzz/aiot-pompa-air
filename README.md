# 💧 Sistem Prediksi Kerusakan Pompa Air Berbasis AIoT
### Pendekatan Klasifikasi Random Forest pada Data Konsumsi Energi

> **Sifah Nur Rizkiyah — 237006035**  
> Universitas Siliwangi | Mata Kuliah: Internet of Things

---

## 📌 Deskripsi Proyek

Sistem AIoT (Artificial Intelligence of Things) untuk memprediksi kondisi pompa air secara real-time menggunakan algoritma **Random Forest Classifier** yang dilatih pada dataset konsumsi energi rumah tangga (*Energydata Complete*).

Sistem ini mengimplementasikan arsitektur **3-layer IoT**:
- **Layer 1 (Perception)** — Sensor ESP32 + DHT22 membaca data lingkungan
- **Layer 2 (Edge Processing)** — Model Random Forest berjalan di edge device
- **Layer 3 (Application)** — Dashboard Streamlit + sistem notifikasi alert

---

## 🚀 Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

🔗 Simulasi Hardware: [Wokwi Project](https://wokwi.com/projects/464319240845028353)

---

## 📁 Struktur File

```
aiot-pompa-air/
├── app.py                    # Dashboard Streamlit utama
├── requirements.txt          # Dependencies Python
├── README.md                 # Dokumentasi ini
├── model_rf_pompa.pkl        # Model Random Forest terlatih
├── scaler_pompa.pkl          # Standard Scaler
├── energydata_complete.csv   # Dataset (19,735 baris)
├── feature_importance.csv    # Importance score setiap fitur
├── project_pompa_aiot.py     # Script training model
├── iot_monitoring.py         # Simulasi IoT monitoring
├── sketch.ino                # Kode Arduino ESP32
├── diagram.json              # Diagram Wokwi
└── libraries.txt             # Library Arduino
```

---

## 🤖 Performa Model

| Metrik | Nilai |
|--------|-------|
| Akurasi | **87.3%** |
| AUC-ROC | **0.935** |
| Precision Normal | 88.7% |
| Recall Normal | 97.2% |
| F1-Score Normal | 92.7% |
| Precision Bermasalah | 84.1% |
| Recall Bermasalah | 54.9% |
| F1-Score Bermasalah | 66.4% |

### Top 5 Feature Importance
| Fitur | Importance |
|-------|-----------|
| lights | 6.25% |
| RH_1 | 6.09% |
| T2 | 5.86% |
| RH_8 | 5.20% |
| RH_out | 4.84% |

---

## ⚙️ Hardware (Wokwi Simulation)

| Pin ESP32 | Komponen | Fungsi |
|-----------|----------|--------|
| D4 | DHT22 | Data suhu & kelembaban |
| D34 | Potensiometer | Simulasi beban motor |
| D21 | OLED SSD1306 | Display SDA |
| D22 | OLED SSD1306 | Display SCL |
| D25 | LED Hijau | Status Normal |
| D26 | LED Kuning | Status Warning |
| D27 | LED Merah | Status Bermasalah |

---

## 🛠️ Cara Menjalankan Lokal

```bash
# 1. Clone repository
git clone https://github.com/username/aiot-pompa-air.git
cd aiot-pompa-air

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pastikan file model tersedia
# model_rf_pompa.pkl, scaler_pompa.pkl, energydata_complete.csv

# 4. Jalankan dashboard
streamlit run app.py
```

---

## ☁️ Deploy ke Streamlit Cloud

1. Push repo ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo
4. Set **Main file path**: `app.py`
5. Klik **Deploy**

> ⚠️ Pastikan semua file `.pkl` dan `.csv` ikut ter-push ke GitHub (ukuran < 100MB per file)

---

## 📡 Topik MQTT (Implementasi Nyata)

```
pompa/sensor    → data mentah sensor
pompa/status    → hasil prediksi RF
pompa/alert     → notifikasi kondisi bermasalah
```

---

## 🔗 Referensi

- Dataset: [Energydata Complete](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- Simulasi: [Wokwi Project 464319240845028353](https://wokwi.com/projects/464319240845028353)
- Algoritma: [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

---

<div align="center">
  💧 <strong>Sifah Nur Rizkiyah — 237006035</strong><br>
  Universitas Siliwangi | Internet of Things 2024
</div>
