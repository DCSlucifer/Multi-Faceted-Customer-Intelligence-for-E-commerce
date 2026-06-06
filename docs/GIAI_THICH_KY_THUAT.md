# Giải thích kỹ thuật & Phân tích kết quả

**Đồ án Data Mining 2026 — Nhóm 4**
*Multi-Faceted Customer Intelligence for E-commerce (bộ dữ liệu Online Retail II)*

Tài liệu này giải thích **từng kỹ thuật đã sử dụng**, **ý nghĩa của output** và **phân tích/nhận định** rút ra từ output đó. Mọi con số đều lấy trực tiếp từ artifacts do pipeline sinh ra (`reports/*.csv`, `reports/manifest.json`), không phải số minh hoạ.

> Cấu hình chung: seed cố định `42`, mốc cắt churn `2011-09-01`, cửa sổ churn `90 ngày`. Phiên bản: Python 3.12.4, pandas 2.2.3, scikit-learn 1.5.2, torch 2.3.1+cpu, shap 0.46.0.

---

## Mục lục
1. [Tiền xử lý & Feature Engineering](#1-tiền-xử-lý--feature-engineering)
2. [Phân cụm khách hàng (Clustering)](#2-phân-cụm-khách-hàng-clustering)
3. [Phân loại / Dự đoán churn (Classification)](#3-phân-loại--dự-đoán-churn-classification)
4. [Đánh giá trung thực: Baseline, Ablation, Calibration](#4-đánh-giá-trung-thực-baseline-ablation-calibration)
5. [Luật kết hợp / Market Basket (Association Rules)](#5-luật-kết-hợp--market-basket-association-rules)
6. [Deep Learning: MLP & Autoencoder phát hiện bất thường](#6-deep-learning-mlp--autoencoder)
7. [Giải thích mô hình (SHAP)](#7-giải-thích-mô-hình-shap)
8. [Tổng hợp liên nhiệm vụ & Tầng quyết định kinh doanh](#8-tổng-hợp-liên-nhiệm-vụ--tầng-quyết-định)
9. [Kết luận tổng quát](#9-kết-luận-tổng-quát)

---

## 1. Tiền xử lý & Feature Engineering

### Kỹ thuật đã dùng
Hàm `clean()` trong [src/preprocessing.py](../src/preprocessing.py) thực hiện:
- **Loại bản ghi thiếu `CustomerID`** — không gán được hành vi cho khách thì bỏ.
- **Bỏ hoá đơn huỷ** (mã `Invoice` bắt đầu bằng `C`) — đây là giao dịch trả hàng, không phải mua thật.
- **Bỏ `Quantity ≤ 0` và `Price ≤ 0`** — loại số lượng/giá âm hoặc bằng 0 (lỗi nhập, hàng tặng, điều chỉnh).
- **Winsorize ở phân vị (1%, 99%)** cho `Quantity` và `Price` — cắt ngọn giá trị cực đoan để outlier không bóp méo trung bình/khoảng cách.
- Tạo `TotalAmount = Quantity × Price`.

**Đặc trưng RFM** (`make_rfm`): `Recency` (số ngày kể từ lần mua cuối tới mốc snapshot), `Frequency` (số hoá đơn khác nhau), `Monetary` (tổng chi tiêu).

**Đặc trưng hành vi bổ sung** dùng cho phân loại: `avg_basket_value` (giá trị giỏ trung bình), `avg_basket_size` (số món/giỏ), `unique_products` (số mã sản phẩm khác nhau), `DominantCountry` (quốc gia mua nhiều nhất).

### Output
| Chỉ số | Giá trị |
|--------|---------|
| Số dòng sau làm sạch | **805,549** (từ 1,067,371 dòng thô) |
| Số khách hàng | **5,878** · 41 quốc gia |
| Số khách được gán nhãn churn | **5,249** |
| Tỉ lệ churn | **57.3%** |

### Phân tích
- Pipeline loại ~24.5% số dòng — phần lớn là bản ghi thiếu CustomerID và hoá đơn huỷ. Tỉ lệ này hợp lý với bộ Online Retail II (nổi tiếng nhiều null CustomerID).
- **5,878 khách nhưng chỉ 5,249 được gán nhãn**: khác biệt là do nhãn churn chỉ tính cho khách **có giao dịch trước mốc cắt** `2011-09-01`. Đây chính là điểm **chống rò rỉ nhãn (leakage-free)**: đặc trưng được tính *chỉ* từ dữ liệu trước mốc cắt, còn nhãn = "không mua trong 90 ngày kế tiếp".
- Tỉ lệ churn 57.3% là **mất cân bằng nhẹ nghiêng về lớp churn** → cần baseline majority-class để đánh giá công bằng (xem mục 4) và SMOTE khi huấn luyện MLP.

---

## 2. Phân cụm khách hàng (Clustering)

### Kỹ thuật đã dùng
Phân cụm trên RFM đã chuẩn hoá (`StandardScaler`) với 3 thuật toán + kiểm định:
- **K-means** — quét `k` và chọn theo silhouette.
- **DBSCAN** — phân cụm theo mật độ, `eps` ước lượng bằng đồ thị k-distance.
- **AGNES** (Agglomerative, linkage `ward`) — phân cụm phân cấp.
- **Chỉ số kiểm định**: Silhouette (càng gần 1 càng tốt), Davies-Bouldin (càng nhỏ càng tốt), Calinski-Harabasz (càng lớn càng tốt).
- **Kiểm tra độ ổn định**: chạy lại K-means với 10 seed khác nhau.

### Output
**`clustering_validation.csv`**

| Thuật toán | n_clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|-----------|-----------|-----------|----------------|-------------------|
| **K-means** | **2** | **0.4193** | 0.8865 | 5840.0 |
| DBSCAN | 13 | **−0.1205** | 1.6319 | 741.9 (403 điểm nhiễu) |
| AGNES | 2 | 0.3878 | 0.7887 | 4303.9 |

**`cluster_stability.csv`**: qua 10 seed, silhouette = 0.4193 với **độ lệch chuẩn = 0.0000**; kích thước cụm ~3189/2689.

**`cluster_profile_kmeans.csv`** (chân dung cụm):

| Cụm | Số KH | Recency TB | Frequency TB | Monetary TB | Tỉ trọng |
|-----|-------|-----------|--------------|-------------|----------|
| 0 | 3,189 | 317.5 ngày | 1.85 | 495.6 | 54.3% |
| 1 | 2,689 | 62.6 ngày | 11.55 | 5,206.1 | 45.7% |

### Phân tích
- **K-means k=2 là lựa chọn tốt nhất**: silhouette cao nhất (0.419) và cao hơn AGNES (0.388). DBSCAN **thất bại** (silhouette âm −0.12, 13 cụm rời rạc + nhiều nhiễu) → dữ liệu RFM không có cấu trúc mật độ rõ, mà phân tách kiểu cầu/khoảng cách → K-means phù hợp hơn.
- **Độ ổn định tuyệt đối** (std silhouette = 0.0000 qua 10 seed) → kết quả không phải may rủi theo khởi tạo, rất đáng tin để báo cáo.
- **Ý nghĩa kinh doanh 2 cụm rõ rệt**:
  - **Cụm 0 — khách "ngủ đông" / giá trị thấp**: lâu không mua (≈317 ngày), mua rất ít (1.85 lần), chi ít (≈496). Chiếm hơn nửa danh sách.
  - **Cụm 1 — khách "đang hoạt động" / giá trị cao**: mới mua gần đây (≈63 ngày), mua thường xuyên (≈11.5 lần), chi cao gấp ~10 lần (≈5,206).
- Silhouette 0.42 là mức **"cấu trúc trung bình–khá"** (không phải tách hoàn hảo), điều này hợp lý vì hành vi khách hàng là liên tục chứ không rời rạc tuyệt đối — vẫn đủ tốt để hành động.

---

## 3. Phân loại / Dự đoán churn (Classification)

### Kỹ thuật đã dùng
Dự đoán khách có churn trong 90 ngày tới hay không. [src/classification.py](../src/classification.py):
- **6 mô hình cổ điển**: LogisticRegression, RandomForest, KNN, GaussianNB, SVM (RBF), DecisionTree.
- **Pipeline chống rò rỉ**: `OneHotEncoder` (gộp nhóm hiếm cho `DominantCountry`) + scaler nằm *bên trong* pipeline CV → chỉ fit trên fold huấn luyện.
- **GridSearchCV** tinh chỉnh siêu tham số, đánh giá bằng **AUC**.
- Báo cáo cả **CV AUC** (trung bình + độ lệch chuẩn) lẫn **AUC trên tập test** tách riêng.

### Output — `classical_results.csv`

| Mô hình | CV AUC (±std) | Test AUC | F1 | Precision | Recall | Accuracy | Tham số tốt nhất |
|---------|---------------|----------|-----|-----------|--------|----------|------------------|
| **LogisticRegression** | 0.799 ±0.016 | **0.785** | 0.739 | 0.770 | 0.711 | 0.712 | C=0.1 |
| RandomForest | 0.800 ±0.018 | 0.782 | 0.745 | 0.747 | 0.744 | 0.709 | max_depth=10, min_leaf=5 |
| KNN | 0.779 ±0.018 | 0.768 | 0.721 | 0.740 | 0.703 | 0.688 | k=21 |
| GaussianNB | 0.774 ±0.015 | 0.760 | 0.744 | 0.724 | 0.766 | 0.698 | — |
| SVM (RBF) | 0.785 ±0.020 | 0.760 | 0.745 | 0.753 | 0.738 | 0.711 | C=1, gamma=scale |
| DecisionTree | 0.775 ±0.012 | 0.748 | 0.724 | 0.754 | 0.696 | 0.695 | max_depth=6, min_leaf=5 |

### Phân tích
- **LogisticRegression thắng trên test (AUC 0.785)**, sát ngay sau là RandomForest (0.782). Một mô hình tuyến tính đơn giản ngang ngửa mô hình phi tuyến phức tạp → **quan hệ giữa đặc trưng và churn gần như tuyến tính**, không cần mô hình quá phức tạp (nguyên tắc Occam → chọn mô hình đơn giản, dễ giải thích).
- **CV AUC ≈ Test AUC** ở mọi mô hình (chênh chỉ ~0.01–0.02) → **không overfit**, mô hình tổng quát hoá tốt.
- `C=0.1` (regularization mạnh) được chọn cho LR → mô hình ưu tiên đơn giản, củng cố nhận định quan hệ tuyến tính + chống overfit.
- DecisionTree đơn lẻ yếu nhất (0.748) nhưng RandomForest (ensemble nhiều cây) vượt hẳn → ensemble giảm phương sai hiệu quả.
- AUC ~0.78 là mức **"khá tốt"** cho bài toán churn thực tế (churn vốn nhiễu, phụ thuộc yếu tố ngoài dữ liệu giao dịch). Quan trọng hơn: con số này được **kiểm chứng trung thực** ở mục 4.

---

## 4. Đánh giá trung thực: Baseline, Ablation, Calibration

Đây là phần thể hiện tính học thuật nghiêm túc: không chỉ khoe AUC mà **chứng minh mô hình thực sự học được tín hiệu**.

### 4.1 Baseline — `baseline_results.csv`
| Baseline | n_features | AUC |
|----------|-----------|-----|
| RandomForest (đủ feature) | 7 | 0.766 |
| LR bỏ Recency | 6 | 0.755 |
| Chỉ dùng Recency | 1 | 0.751 |
| Đoán theo lớp đa số | 1 | **0.500** |

**Phân tích**: Mô hình đầy đủ (0.785) > chỉ-Recency (0.751) > đoán mù (0.50). Khoảng cách so với majority-class chứng minh mô hình **thực sự có giá trị**. Nhưng chỉ riêng Recency đã đạt 0.751 → **Recency là yếu tố chi phối**, các đặc trưng còn lại chỉ thêm ~0.034 AUC.

### 4.2 Ablation (cắt bỏ nhóm đặc trưng) — `ablation_results.csv`
| Nhóm đặc trưng | n | Test AUC |
|----------------|---|----------|
| RFM_only | 3 | 0.784 |
| Behavioral_only | 3 | 0.720 |
| RFM + Behavioral | 6 | **0.787** |
| RFM + Behavioral + Country | 7 | 0.784 |
| Without_Recency | 6 | 0.755 |

**Phân tích**:
- **RFM là xương sống** (0.784 chỉ với 3 đặc trưng). Đặc trưng hành vi đơn lẻ yếu hơn (0.720).
- Kết hợp RFM + Behavioral đạt đỉnh (0.787); **thêm Country lại giảm nhẹ** (0.784) → quốc gia gần như không mang thêm tín hiệu, thậm chí thêm nhiễu.
- Bỏ Recency tụt xuống 0.755 → **Recency là đặc trưng quan trọng nhất**, nhất quán với baseline và SHAP (mục 7).

### 4.3 Calibration — `calibration_results.csv`
- **Brier score = 0.189** (càng nhỏ càng tốt).
- Đường reliability (10 bin): `mean_predicted_prob` ≈ `fraction_positive` ở mọi bin (vd bin 1: dự đoán 0.10 ↔ thực tế 0.105; bin 10: dự đoán 0.913 ↔ thực tế 0.933).

**Phân tích**: Xác suất mô hình đưa ra **đáng tin về mặt định lượng** — khi mô hình nói "70% churn" thì thực tế ~70% khách đó churn thật. Điều này cực kỳ quan trọng vì tầng quyết định kinh doanh (mục 8) dùng trực tiếp xác suất churn để phân nhóm hành động.

> **Tóm lại mục 4**: AUC 0.78 là **thật và trung thực** — vượt rõ baseline, được giải thích bằng ablation, và xác suất đã được hiệu chuẩn tốt.

---

## 5. Luật kết hợp / Market Basket (Association Rules)

### Kỹ thuật đã dùng
[src/association.py](../src/association.py): mã hoá one-hot giỏ hàng theo hoá đơn, khai phá tập phổ biến bằng **Apriori** và **FP-Growth**, sinh luật và lọc theo **lift ≥ 1.5**. `min_support = 0.02`.

- **Support**: tần suất xuất hiện đồng thời.
- **Confidence**: P(vế phải | vế trái).
- **Lift**: mức độ mua kèm cao hơn ngẫu nhiên (lift > 1 = tương quan dương).

### Output — `apriori_vs_fpgrowth.csv` & `association_rules.csv`
| Thuật toán | Tập phổ biến | Luật (lift≥1.5) | Thời gian (s) |
|-----------|--------------|------------------|----------------|
| Apriori | 242 | 17 | 7.147 |
| FP-Growth | 242 | 17 | **4.249** |

**Top luật (theo lift):**
| Vế trái → Vế phải | Support | Confidence | Lift |
|-------------------|---------|-----------|------|
| GREEN REGENCY TEACUP → ROSES REGENCY TEACUP | 0.021 | **0.797** | **26.76** |
| ROSES REGENCY TEACUP → GREEN REGENCY TEACUP | 0.021 | 0.705 | 26.76 |
| SWEETHEART TRINKET BOX → STRAWBERRY TRINKET BOX | 0.024 | 0.732 | 13.89 |
| WOODEN PICTURE FRAME WHITE → WOODEN FRAME ANTIQUE WHITE | 0.028 | 0.599 | 12.13 |
| JUMBO BAG STRAWBERRY → JUMBO BAG RED WHITE SPOTTY | 0.027 | 0.629 | 6.93 |

### Phân tích
- **FP-Growth nhanh hơn Apriori ~1.4×** (4.25s vs 7.15s) trên cùng kết quả (242 tập, 17 luật) → minh hoạ đúng lý thuyết: FP-Growth tránh sinh ứng viên nên nhanh hơn khi dữ liệu lớn.
- **Lift cực cao (26.8)** ở bộ tách trà Regency: khách mua cốc xanh thì 79.7% mua kèm cốc hồng. Đây là **bộ sản phẩm cùng dòng/cùng set** → gợi ý bán combo, gợi ý "mua kèm".
- Các luật còn lại đều là **sản phẩm cùng họ** (khung ảnh gỗ, hộp trinket, túi jumbo, khay nến) → khách có xu hướng sưu tập trọn bộ. Hành động: **bundle, cross-sell, sắp xếp trưng bày cạnh nhau**.
- Support nhỏ (~2-3%) nhưng lift rất cao → luật **hiếm nhưng cực mạnh**, lý tưởng cho gợi ý cá nhân hoá thay vì khuyến mãi đại trà.

---

## 6. Deep Learning: MLP & Autoencoder

### 6.1 MLP cho dự đoán churn
**Kiến trúc** (`build_mlp`): `Linear(in→64) → ReLU → Dropout(0.3) → Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→16) → ReLU → Linear(16→1)`. Tối ưu Adam (lr=1e-3), early stopping (patience=8), **SMOTE chỉ trên tập train** để cân bằng lớp.

**Output — `mlp_vs_classical.csv`**:
| Mô hình | AUC | F1 | Precision | Recall |
|---------|-----|-----|-----------|--------|
| LogisticRegression | **0.785** | 0.739 | 0.770 | 0.711 |
| MLP | 0.783 | 0.738 | 0.756 | 0.721 |

**Phân tích**: MLP (0.783) **không vượt** LogisticRegression (0.785). Đây là kết quả **trung thực và đáng giá về mặt học thuật**: với dữ liệu dạng bảng, ít đặc trưng (7) và quan hệ gần tuyến tính, **deep learning không có lợi thế** — thậm chí thua nhẹ mô hình tuyến tính. Kết luận: chọn LogisticRegression vì đơn giản, nhanh, dễ giải thích, hiệu năng tương đương.

### 6.2 Autoencoder phát hiện bất thường
**Kiến trúc** (`build_autoencoder`): Encoder `in→16→8→4` (bottleneck=4), Decoder `4→8→16→in`. Huấn luyện tái tạo đặc trưng; khách có **lỗi tái tạo (reconstruction error) cao** = bất thường. Lấy top 5%.

**Output**: `autoencoder_final_loss = 0.097`, **263 khách hàng** bị gắn cờ bất thường (top 5%) → `reports/anomaly_customers.csv`.

**Phân tích**:
- Loss hội tụ thấp (0.097) → autoencoder học tốt mẫu hành vi "bình thường".
- 263 khách (5%) có hành vi lệch khỏi số đông — thường là khách chi tiêu cực lớn, mua bất thường nhiều, hoặc mẫu mua kỳ lạ. Trong `customer_decision_summary.csv`, phần lớn anomaly tập trung ở **nhóm VIP (180/263)** → khách giá trị cao thường "bất thường" theo nghĩa tốt; số ít rơi vào nhóm cần **review thủ công** (nghi ngờ gian lận/lỗi dữ liệu).

---

## 7. Giải thích mô hình (SHAP)

### Kỹ thuật
Áp dụng SHAP trên **cả RandomForest** (`TreeExplainer`) **và MLP** (`KernelExplainer`) để xác định đặc trưng nào đẩy dự đoán churn. Hình: `reports/figures/05_shap_*`.

### Phân tích
- Cả hai mô hình **đồng thuận**: **Recency là yếu tố churn mạnh nhất** (lâu không mua → khả năng churn cao), tiếp theo là các đặc trưng hành vi/Frequency.
- Sự đồng thuận giữa một mô hình cây và một mạng nơ-ron **củng cố độ tin cậy của giải thích** — không phải artefact của một loại mô hình.
- Nhất quán hoàn toàn với baseline (chỉ-Recency đạt 0.751) và ablation (bỏ Recency tụt 0.03) ở mục 4 → **3 phương pháp độc lập cùng chỉ ra Recency**.

---

## 8. Tổng hợp liên nhiệm vụ & Tầng quyết định

### 8.1 Churn theo cụm — `cluster_churn_rate.csv` / `churn_by_cluster.csv`
| Cụm | Số KH | Tỉ lệ churn | Recency TB | Frequency TB | Monetary TB |
|-----|-------|------------|-----------|--------------|-------------|
| 0 (ngủ đông) | 2,697 | **90.0%** | 367.9 | 1.96 | 519.5 |
| 1 (hoạt động) | 2,552 | **23.0%** | 65.3 | 11.97 | 5,401.2 |

**Phân tích — đây là phát hiện quan trọng nhất của đồ án**:
- **Khoảng cách churn 67.3 điểm phần trăm** giữa hai phân khúc (90% vs 23%). Phân cụm không giám sát (RFM) và dự đoán churn có giám sát (nhãn) **xác nhận chéo lẫn nhau**: cụm "ngủ đông" gần như chắc chắn rời bỏ, cụm "hoạt động" gắn bó cao.
- Điều này biến phân khúc thành **công cụ hành động trực tiếp**: dồn ngân sách giữ chân vào cụm 0, dồn ngân sách bán thêm (cross-sell) vào cụm 1.

### 8.2 Tầng quyết định kinh doanh — `customer_decision_summary.csv`
Kết hợp **xác suất churn × giá trị khách × cờ bất thường × phân khúc** để gán hành động cho **toàn bộ 5,249 khách**:

| Hành động đề xuất | Số khách |
|-------------------|----------|
| Mục tiêu giữ chân (retention) | **1,620** |
| Mục tiêu bán thêm (cross-sell) | **1,154** |
| Review thủ công | **61** |
| Còn lại | Monitor / nurture tiêu chuẩn |

Ví dụ các nhóm cụ thể: *VIP nurture/Cross-sell* (1,153 khách, churn TB 0.18, Monetary TB 8,124), *Win-back campaign* cho nhóm churn cao giá trị thấp (930 khách, churn TB 0.85)...

**Phân tích**:
- Pipeline không dừng ở mô hình mà **kết tinh thành quyết định kinh doanh có thể thực thi** — đúng tinh thần "data mining phục vụ ra quyết định".
- Logic nhất quán: nhóm Win-back có churn TB rất cao (0.80–0.85) + giá trị thấp; nhóm VIP có churn thấp (0.18) + giá trị cao. Xác suất churn dùng ở đây **đáng tin nhờ đã calibrate** (mục 4.3).
- 61 khách "review thủ công" chủ yếu trùng với cờ anomaly → tách riêng cho con người kiểm tra thay vì tự động hoá mù quáng.

---

## 9. Kết luận tổng quát

| Trụ cột | Bằng chứng |
|---------|-----------|
| **Chống rò rỉ nhãn** | Đặc trưng chỉ từ dữ liệu trước mốc cắt; encoder/scaler trong pipeline CV; nhãn 90 ngày tương lai |
| **Trung thực** | AUC 0.785 vượt baseline (0.50/0.75), giải thích bằng ablation, xác suất đã calibrate (Brier 0.189) |
| **Tái lập** | Seed 42, version pinned, `manifest.json` ghi provenance, pipeline một lệnh |
| **Nhất quán chéo** | Recency là driver churn theo baseline + ablation + SHAP (RF & MLP) |
| **Giá trị kinh doanh** | Khoảng cách churn 67.3đ giữa 2 phân khúc → 5,249 khách được gán hành động cụ thể |
| **Khiêm tốn khoa học** | DL (MLP/AE) được thử nhưng thừa nhận không vượt mô hình cổ điển trên dữ liệu bảng |

**Thông điệp chính**: Đồ án không chạy theo việc "ép cho ra số đẹp" mà xây dựng một pipeline **trung thực, tái lập được, kiểm chứng chéo** — từ làm sạch dữ liệu đến quyết định kinh doanh — trong đó mỗi kết luận đều được ít nhất hai phương pháp độc lập xác nhận.

---

*Mọi số liệu trong tài liệu này được tái sinh tự động bởi pipeline và ghi trong `reports/manifest.json`. Xem `README.md` để chạy lại toàn bộ.*
