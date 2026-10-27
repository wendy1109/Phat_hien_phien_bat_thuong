# Dự án: Phát Hiện Phiên Bất Thường (Session Anomaly Detection)

Dự án áp dụng các phương pháp thống kê và học máy (Machine Learning) để phân tích và phát hiện các phiên truy cập bất thường từ tập dữ liệu log/giám sát hệ thống.

---

## 📌 Các tính năng chính

1. **Đọc & tiền xử lý dữ liệu (`du_lieu_tho.csv`)**:
   - Loại bỏ các bản ghi không hợp lệ hoặc có giá trị âm.
   - Điền khuyết (imputation) bằng giá trị trung vị (median).
   - Loại bỏ các dòng trùng lặp dựa trên tập thuộc tính đặc trưng.

2. **Thống kê mô tả & Phân tích tương quan**:
   - Tính toán trung bình, độ lệch chuẩn, phương sai, phân vị,...
   - Tính toán ma trận tương quan giữa các thuộc tính.

3. **Phát hiện phiên bất thường bằng 3 phương pháp**:
   - **IQR (Interquartile Range)**: Dựa trên khoảng tứ phân vị.
   - **Z-Score**: Chuẩn hóa `StandardScaler` và gắn cờ các phiên vượt ngưỡng $|Z| > 3$.
   - **K-Means Clustering**: Phân nhóm dữ liệu thành các cụm (k=2) và xác định cụm có tỷ lệ bất thường cao.

4. **Đánh giá & Trực quan hóa kết quả**:
   - Biểu đồ 1: So sánh số lượng phiên bất thường phát hiện được giữa các phương pháp.
   - Biểu đồ 2: Phân phối thời lượng phiên truy cập.
   - Biểu đồ 3: Heatmap ma trận tương quan giữa các đặc trưng.
   - Xuất dữ liệu thống kê, ma trận tương quan và kết quả gán nhãn vào thư mục `ket_qua_bat_thuong/`.

---

## 📂 Cấu trúc thư mục

```text
├── BTLpython.py                  # Script chính thực thi tiền xử lý, mô hình và trực quan hóa
├── du_lieu_tho.csv              # File dữ liệu đầu vào
├── ket_qua_bat_thuong/          # Thư mục lưu kết quả xuất ra
│   ├── bieu_do_1.png            # Biểu đồ so sánh các phương pháp
│   ├── bieu_do_2.png            # Biểu đồ phân phối thời lượng phiên
│   ├── bieu_do_3.png            # Heatmap tương quan
│   ├── ket_qua_phat_hien.csv    # Kết quả phát hiện bất thường của từng phiên
│   ├── ma_tran_tuong_quan.csv   # Ma trận hệ số tương quan
│   └── thong_ke.csv             # Bảng thống kê mô tả
├── .gitignore                   # Cấu hình bỏ qua các file rác / môi trường ảo
└── README.md                    # Tài liệu hướng dẫn dự án
```

---

## 🚀 Hướng dẫn cài đặt & Chạy chương trình

### 1. Cài đặt các thư viện phụ thuộc

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 2. Chạy chương trình

```bash
python BTLpython.py
```
