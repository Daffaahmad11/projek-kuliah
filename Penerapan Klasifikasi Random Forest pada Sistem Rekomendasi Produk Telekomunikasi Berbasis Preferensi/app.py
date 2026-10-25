from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def force_integer_string(x):
    try:
        val = float(x)
        if val.is_integer():
            return str(int(val))
        return str(val)
    except:
        return str(x)

def format_rupiah(x):
    try:
        return f"Rp {int(float(x)):,}".replace(',', '.')
    except:
        return x

try:
    data_otak = joblib.load('model_telkomsel.pkl')
    model_rf = data_otak['model_rf']
    le = data_otak['encoder']
    
    df_paket_display = pd.read_csv('dataoutput.csv', header=1)
    
    df_paket_display.dropna(subset=['nama_paket', 'harga_rupiah', 'kuota_gb'], inplace=True)
    
    df_paket_display['harga_formatted'] = df_paket_display['harga_rupiah'].apply(format_rupiah)
    
    df_paket_display['kuota_gb'] = df_paket_display['kuota_gb'].apply(force_integer_string)
    
    print(" System Ready: Database & AI Loaded")

except Exception as e:
    print(f" ERROR LOAD MODEL: {e}")
    df_paket_display = pd.DataFrame() 

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/dashboard', methods=['POST'])
def dashboard_page():
    nama_user = request.form.get('nama')
    no_hp_user = request.form.get('no_hp')
    
    list_paket = df_paket_display.to_dict(orient='records')
    
    return render_template('dashboard.html', 
                           nama=nama_user, 
                           no_hp=no_hp_user, 
                           paket_tersedia=list_paket)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        df_paket_ai = data_otak['database_paket']
        
        data = request.json
        quota_input = float(data['quota'])
        budget_input = float(data['budget'])

        input_df = pd.DataFrame([[quota_input, budget_input]], columns=['quota_clean', 'budget_clean'])

        pred_idx = model_rf.predict(input_df)[0]
        kategori_prediksi = le.inverse_transform([pred_idx])[0]
        
        if quota_input <= 5 and kategori_prediksi in ['Bulanan', 'Tahunan']:
            kategori_prediksi = 'Harian'
        elif quota_input <= 15 and kategori_prediksi == 'Tahunan':
            kategori_prediksi = 'Bulanan'

        calon_paket = df_paket_ai[df_paket_ai['kategori_paket'] == kategori_prediksi].copy()
        if len(calon_paket) == 0: calon_paket = df_paket_ai.copy()

        scaler = MinMaxScaler()
        X_paket = calon_paket[['quota_clean', 'price_clean']].values
        X_user = np.array([[quota_input, budget_input]])
        
        scaler.fit(X_paket)
        X_paket_scaled = scaler.transform(X_paket)
        X_user_scaled = scaler.transform(X_user)
        
        k = min(3, len(calon_paket))
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(X_paket_scaled)
        distances, indices = knn.kneighbors(X_user_scaled)
        
        best_idx = indices[0][0]
        best_paket = calon_paket.iloc[best_idx]
        
        harga_real = float(best_paket['price_clean'])
        kuota_real = float(best_paket['quota_clean'])
        
        if harga_real <= budget_input:
            score_harga = 100
        else:
            selisih = harga_real - budget_input
            score_harga = 100 * (1 - (selisih / harga_real))
        
        pembagi_kuota = max(quota_input, kuota_real)
        if pembagi_kuota == 0: score_kuota = 0
        else: score_kuota = 100 * (1 - (abs(quota_input - kuota_real) / pembagi_kuota))

        final_score = (score_harga * 0.6) + (score_kuota * 0.4)
        final_score = int(min(max(final_score, 10), 99))
        
        if abs(harga_real - budget_input) < 5000 and abs(kuota_real - quota_input) < 1:
            final_score = 99

        hasil_rekomendasi = calon_paket.iloc[indices[0]]
        rekomendasi_list = []
        
        for _, row in hasil_rekomendasi.iterrows():
            q_val = float(row['quota_clean'])
            
            if q_val.is_integer():
                q_str = str(int(q_val))
            else:
                q_str = str(q_val)

            rekomendasi_list.append({
                'nama': row['nama_paket'],
                'kuota': q_str,
                'harga': f"Rp {int(row['price_clean']):,}".replace(',', '.'),
                'masa_aktif': int(row['masa_aktif_hari'])
            })

        if final_score >= 90: pesan = "Sempurna! Paket ini sangat pas dengan keinginanmu."
        elif final_score >= 70: pesan = "Cukup akurat, mendekati budget & kuota yang diminta."
        elif final_score >= 50: pesan = "Opsi terbaik yang tersedia, meski agak berbeda."
        else:
            kategori_prediksi = "Alternatif Terdekat"
            pesan = "Permintaan terlalu jauh dari paket yang tersedia di database."

        return jsonify({
            'status': 'success',
            'kategori_user': kategori_prediksi,
            'confidence': final_score,
            'rekomendasi': rekomendasi_list,
            'pesan_khusus': pesan
        })

    except Exception as e:
        print(f"Error pada prediksi: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)