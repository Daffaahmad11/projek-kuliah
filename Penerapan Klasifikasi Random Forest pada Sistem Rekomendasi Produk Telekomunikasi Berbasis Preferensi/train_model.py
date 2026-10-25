import pandas as pd
import numpy as np
import re
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

FILE_LATIH = "datalatih.xlsx" 
FILE_PAKET = "dataoutput.csv"

print(" Memulai Training Model: SYSTEM START...")

def clean_money(x):
    if pd.isna(x): return 0
    x = str(x).lower().replace('rp','').replace('.','').replace(',','').replace(' ','')
    if 'k' in x: x = x.replace('k','000')
    try: return float(re.findall(r"[\d]+", x)[0])
    except: return 0

def clean_quota(x):
    if pd.isna(x): return 0
    x = str(x).lower().replace('gb','').replace(' ','')
    if 'unlimited' in x: return 999
    try:
        if '-' in x: a, b = x.split('-'); return (float(a)+float(b))/2
        return float(re.findall(r"[\d\.]+", x)[0])
    except: return 0

def tag_paket(days):
    try:
        days = float(days)
        if days <= 3: return 'Harian'
        elif days <= 14: return 'Mingguan'
        elif days <= 60: return 'Bulanan'
        else: return 'Tahunan'
    except: return 'Lainnya'

print(" Loading Dataset...")
try:
    df_train = pd.read_excel(FILE_LATIH)
except:
    df_train = pd.read_csv(FILE_LATIH)

df_paket = pd.read_csv(FILE_PAKET, header=1)

df_train['quota_clean'] = df_train.iloc[:, 5].apply(clean_quota)
df_train['budget_clean'] = df_train.iloc[:, 6].apply(clean_money)
df_train['target_preference'] = df_train.iloc[:, 7].astype(str).str.title().str.strip()

valid_cats = ['Harian', 'Mingguan', 'Bulanan', 'Tahunan']
df_train = df_train[df_train['target_preference'].isin(valid_cats)].copy()

df_paket['quota_clean'] = df_paket['kuota_gb'].apply(clean_quota)
df_paket['price_clean'] = df_paket['harga_rupiah'].apply(clean_money)
df_paket['kategori_paket'] = df_paket['masa_aktif_hari'].apply(tag_paket)

df_paket = df_paket[(df_paket['quota_clean'] > 0) & (df_paket['price_clean'] > 0)].copy()

print(" Melakukan Data Augmentation (Berbasis Statistik Paket Asli)...")

stats_paket = df_paket.groupby('kategori_paket').agg({
    'price_clean': ['min', 'max'],
    'quota_clean': ['min', 'max']
})

dummy_data = []

for kategori in stats_paket.index:
    if kategori not in valid_cats: continue 

    min_price = stats_paket.loc[kategori, ('price_clean', 'min')]
    max_price = stats_paket.loc[kategori, ('price_clean', 'max')]
    min_quota = stats_paket.loc[kategori, ('quota_clean', 'min')]
    max_quota = stats_paket.loc[kategori, ('quota_clean', 'max')]

    lower_factor = 0.8  
    upper_factor = 1.5  
    if kategori == 'Tahunan': upper_factor = 20.0 

    for _ in range(100):
        simulasi_quota = np.random.uniform(min_quota * 0.9, max_quota * 1.1)
        simulasi_budget = np.random.uniform(min_price * lower_factor, max_price * upper_factor)
        
        dummy_data.append({
            'quota_clean': simulasi_quota, 
            'budget_clean': simulasi_budget, 
            'target_preference': kategori
        })

df_add = pd.DataFrame(dummy_data)
df_train_balanced = pd.concat([df_train, df_add], ignore_index=True).ffill()

print(f" Data Augmentation Selesai. Total Data Latih: {len(df_train_balanced)} baris.")

X = df_train_balanced[['quota_clean', 'budget_clean']]
y = df_train_balanced['target_preference']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(" Melatih Random Forest (Navigator Kategori)...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf.fit(X, y_encoded)

otak_sistem = {
    'model_rf': rf, 
    'encoder': le, 
    'database_paket': df_paket 
}

joblib.dump(otak_sistem, 'model_telkomsel.pkl')
print(" SYSTEM READY! File 'model_telkomsel.pkl' berhasil disimpan.")
print("   -> Isolation Forest telah dihapus untuk efisiensi.")