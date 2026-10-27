import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score
)

# 1. DOC DU LIEU
df = pd.read_csv("du_lieu_tho.csv", encoding="utf-8-sig")

cot = [
    "thoi_luong_phien_giay",
    "so_phien_dong_thoi_cung_ip",
    "so_trang_da_xem",
    "so_lan_dang_nhap_that_bai",
    "so_luot_goi_api",
    "so_api_that_bai",
    "thoi_gian_phan_hoi_tb_ms",
    "cpu_may_chu_phan_tram",
    "so_truy_van_cham",
    "so_loi_http_4xx",
    "so_loi_http_5xx"
]

features = [c for c in cot if c in df.columns]

if len(df) < 200 or len(features) < 5:
    raise ValueError("Can it nhat 200 ban ghi va 5 thuoc tinh.")

print("Du lieu ban dau:", df.shape)

# 2. LAM SACH DU LIEU
for c in features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df[c] < 0, c] = float("nan")

    if df[c].notna().sum() == 0:
        raise ValueError(f"Cot {c} khong co du lieu hop le.")

    df[c] = df[c].fillna(df[c].median())

print("Gia tri thieu sau xu ly:", df[features].isna().sum().sum())
print("Dong trung lap:", df.duplicated(subset=features).sum())

df = df.drop_duplicates(subset=features).reset_index(drop=True)
os.makedirs("ket_qua_bat_thuong", exist_ok=True)

# 3. THONG KE MO TA
thong_ke = df[features].describe().T
thong_ke["Phuong sai"] = df[features].var()

print("\nTHONG KE MO TA")
print(thong_ke.round(2))

thong_ke.to_csv(
    "ket_qua_bat_thuong/thong_ke.csv",
    encoding="utf-8-sig"
)

tuong_quan = df[features].corr()

# 4. PHAT HIEN BAT THUONG BANG IQR
iqr_flag = pd.DataFrame(index=df.index)

for c in features:
    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)
    iqr = q3 - q1

    iqr_flag[c] = (
        (df[c] < q1 - 1.5 * iqr) |
        (df[c] > q3 + 1.5 * iqr)
    )

df["IQR"] = iqr_flag.any(axis=1).astype(int)

# 5. PHAT HIEN BAT THUONG BANG Z-SCORE
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

df["Z-score"] = (abs(X) > 3).any(axis=1).astype(int)

# 6. PHAN CUM K-MEANS
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cum"] = kmeans.fit_predict(X)

# Cum co ty le Z-score cao hon duoc xem la cum nghi van
ty_le = df.groupby("Cum")["Z-score"].mean()
cum_nghi_van = ty_le.idxmax()

df["K-Means"] = (df["Cum"] == cum_nghi_van).astype(int)

# 7. TONG HOP KET QUA
print("\nKET QUA PHAT HIEN")
for c in ["IQR", "Z-score", "K-Means"]:
    print(f"{c}: {df[c].sum()} phien bat thuong")

# 8. DANH GIA
if "la_bat_thuong" in df.columns:
    y = pd.to_numeric(df["la_bat_thuong"], errors="coerce")

    if y.notna().all() and y.isin([0, 1]).all():
        for ten, du_doan in [
            ("IQR", df["IQR"]),
            ("Z-score", df["Z-score"]),
            ("K-Means", df["K-Means"])
        ]:
            print(f"\nDanh gia {ten}")
            print("Accuracy:", round(accuracy_score(y, du_doan), 3))
            print("Precision:", round(
                precision_score(y, du_doan, zero_division=0), 3))
            print("Recall:", round(
                recall_score(y, du_doan, zero_division=0), 3))
            print("F1-score:", round(
                f1_score(y, du_doan, zero_division=0), 3))
    else:
        print("Nhan khong hop le, khong tinh duoc cac chi so phan loai.")
else:
    print("\nSilhouette Score:",
          round(silhouette_score(X, df["Cum"]), 3))

# 9. BIEU DO 1: SO SANH CAC PHUONG PHAP
plt.figure(figsize=(7, 4))
plt.bar(
    ["IQR", "Z-score", "K-Means"],
    [df[c].sum() for c in ["IQR", "Z-score", "K-Means"]]
)
plt.title("So phien bi danh dau bat thuong")
plt.ylabel("So phien")
plt.tight_layout()
plt.savefig("ket_qua_bat_thuong/bieu_do_1.png", dpi=300)
plt.show()

# 10. BIEU DO 2: PHAN PHOI THOI LUONG PHIEN
if "thoi_luong_phien_giay" in features:
    plt.figure(figsize=(7, 4))
    plt.hist(df["thoi_luong_phien_giay"], bins=20, edgecolor="black")
    plt.title("Phan phoi thoi luong phien")
    plt.xlabel("Thoi luong (giay)")
    plt.ylabel("So phien")
    plt.tight_layout()
    plt.savefig("ket_qua_bat_thuong/bieu_do_2.png", dpi=300)
    plt.show()

# 11. BIEU DO 3: HEATMAP TUONG QUAN
plt.figure(figsize=(10, 7))
plt.imshow(tuong_quan, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(label="He so tuong quan")
plt.xticks(range(len(features)), features, rotation=90, fontsize=8)
plt.yticks(range(len(features)), features, fontsize=8)
plt.title("Heatmap tuong quan cac thuoc tinh")
plt.tight_layout()
plt.savefig("ket_qua_bat_thuong/bieu_do_3.png", dpi=300)
plt.show()

# 12. LUU KET QUA
df.to_csv(
    "ket_qua_bat_thuong/ket_qua_phat_hien.csv",
    index=False,
    encoding="utf-8-sig"
)

tuong_quan.to_csv(
    "ket_qua_bat_thuong/ma_tran_tuong_quan.csv",
    encoding="utf-8-sig"
)

print("\nDA HOAN THANH!")
print("So phien sau lam sach:", len(df))
print("Ket qua duoc luu trong thu muc ket_qua_bat_thuong.")