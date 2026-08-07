# Evaluating ML-Embed-0.6B on Vietnamese Text Classification (Sentiment, Topic) & Retrieval

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange.svg)](https://huggingface.co/datasets)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Bài toán

Các mô hình Nhúng đa ngữ cần có khả năng biểu diễn ngữ nghĩa chính xác trên nhiều cấp độ văn bản khác nhau — từ bài báo dài có cấu trúc đến các bình luận ngắn, không liền mạch. Tuy nhiên, các đánh giá hiện tại trên tiếng Việt thường chỉ tập trung vào một tác vụ đơn lẻ, thiếu cái nhìn toàn diện về độ bền vững của không gian vector.

Mục tiêu của nghiên cứu này bao gồm hai phần:
*   **(a) Kiểm chứng benchmark MTEB Tiếng Việt:** Đánh giá năng lực truy xuất thông tin của backbone **`ML-Embed-0.6B`** trên bộ tiêu chuẩn MTEB Tiếng Việt giống bài nghiên cứu.
*   **(b) Mở rộng kiểm chứng:** Do các giới hạn thực tế về tài nguyên tính toán khi kiểm thử toàn bộ 50 tác vụ MTEB, nhóm đã mở rộng thực nghiệm sang hai bài toán phân loại đối lập về tính chất: **Phân loại 10 chủ đề báo chí** và **Phân loại 3 nhãn cảm xúc sinh viên**.

---

## 2. Thực nghiệm MTEB Tiếng Việt: Kết quả & Giới hạn

Để đối chứng với các báo cáo trên bảng xếp hạng MTEB đa ngữ, chúng tôi đã triển khai quy trình kiểm thử độc lập cho mô hình `ML-Embed-0.6B`[cite: 6] trên các tác vụ tiếng Việt thuộc bộ tiêu chuẩn MTEB.

### 2.1 Kết quả tái hiện trên 7 tác vụ truy xuất
Nhóm đã trích xuất thành công điểm số đánh giá chuẩn `NDCG@10` (Main Score), `MAP@10` và `Recall@10` cho **7 tập dữ liệu truy xuất thông tin Tiếng Việt**:

| Tác vụ MTEB Tiếng Việt | Chỉ số chính (`NDCG@10`) | `MAP@10` | `Recall@10` | Thời gian chạy (giây) |
| :--- | :---: | :---: | :---: | :---: |
| **SciFact-VN** | **61.82%** | 0.57183 | 0.74415 | ~1,391s |
| **HotpotQA-VN** | **53.43%** | 0.44436 | 0.55498 | ~10,820s |
| **FEVER-VN** | **41.51%** | 0.36698 | 0.53798 | ~14,359s |
| **NQ-VN** | **39.93%** | 0.33602 | 0.57215 | ~7,601s |
| **ArguAna-VN** | **35.45%** | 0.23317 | 0.73900 | ~1,876s |
| **ClimateFEVER-VN** | **6.64%** | 0.04860 | 0.07460 | ~14,100s |
| **DBPedia-VN** | **4.79%** | 0.02322 | 0.03828 | ~9,336s |
| **TRUNG BÌNH** | **~34.80%** | — | — | — |

### 2.2 Lý do không chạy đủ 50 tác vụ & Thay đổi thực nghiệm
Mặc dù bài nghiên cứu gốc cung cấp điểm trung bình trên toàn bộ 50 tác vụ tiếng Việt (`Viet.^(50)`), thực nghiệm của nhóm **không thể hoàn thành trọn vẹn 50 tác vụ này** do các rào cản về phần cứng của **Google Colab bản miễn phí**:

*   **Giới hạn thời gian thực thi:** Gói Colab miễn phí giới hạn phiên hoạt động ~4–6 giờ. Trong khi đó, chỉ riêng một tác vụ truy xuất từ `FEVER-VN` (~14,359s $\approx$ **3.99 giờ**) hay `ClimateFEVER-VN` (~14,100s $\approx$ **3.92 giờ**) đã tiêu tốn gần hết hạn mức. Chạy 50 tác vụ đòi hỏi hàng chục giờ GPU liên tục là không khả thi.

> **Thay đổi thực nhiệm:** 
> Nhóm **quyết định thay đổi hướng kiểm chứng sang 2 bộ dataset tiếng Việt: VNTC và UIT-VSFC**. Mọi phân tích về chất lượng biểu diễn của mô hình từ phần dưới đây sẽ được căn cứ trên 2 bộ dữ liệu này. 

---

## 3. Kiến trúc hệ thống

Pipeline thực nghiệm cho 2 tác vụ phân loại được thiết kế dùng chung, gồm ba khối: `Data Streaming` → `Embedding Extraction (ML-Embed-0.6B)` → `Linear Probing (Scikit-Learn)`. Toàn bộ quá trình **không thay đổi trọng số của backbone**, đảm bảo kết quả phản ánh chuẩn xác chất lượng biểu diễn nguyên bản.

| Bước | Tên | Vai trò kỹ thuật |
| :--- | :--- | :--- |
| 1 | Parquet Streaming | Nạp trực tiếp dữ liệu từ Hugging Face cho cả VNTC và UIT-VSFC|
| 2 | Stratified Sampling | Trộn ngẫu nhiên có cố định seed (`seed=42`), lấy mẫu cân bằng (VNTC: 1.9k Train / 2.5k Test). Đối với dữ liệu VNTC nhóm đã cắt ngắn số lượng dòng dữ liệu của file train và file test để thuận tiện cho việc demo|
| 3 | Dual-Probing | Huấn luyện độc lập 2 bộ phân loại tuyến tính cho Tác vụ Chủ đề và Tác vụ Cảm xúc |

**Các thành phần chính**:

| Thành phần | File Script | Hàm / Biến chính | Nhiệm vụ kỹ thuật |
| :--- | :--- | :--- | :--- |
| Nạp dữ liệu VNTC | `evaluate_vntc.py` | `load_vntc_dataset()` | Stream file `vntc_train/test.parquet`, gán nhãn 0–9 |
| Nạp dữ liệu VSFC | `evaluate_vsfc.py` | `load_vsfc_dataset()` | Stream file `vsfc_train/test.parquet`, gán nhãn 0–2 |
| Subsampling (VNTC) | `evaluate_vntc.py` | `.shuffle(seed=42).select(...)` | Rút gọn thời gian benchmark từ 35 phút $\rightarrow$ 2.5 phút |
| Feature Extraction | Cả hai file | `model.encode(batch_size=32)` | Chuyển chuỗi token thành vector dense chuẩn hóa L2 |
| Evaluator | Cả hai file | `LogisticRegression(C=1.0)` | Đo lường Accuracy, Macro/Weighted F1, Classification Report |

---

## 4. Cách chạy

Toàn bộ pipeline thực nghiệm được thiết kế tối giản để dễ dàng tái hiện trên local GPU hoặc Google Colab/Kaggle:

```bash
# 1. Clone GitHub Repository về máy và di chuyển vào thư mục dự án
git clone (https://github.com/MinhPhucTr/CS2202.CH202.git)
cd YOUR_REPOSITORY_NAME

# 2. Cài đặt PyTorch hỗ trợ CUDA (Khuyến nghị CUDA 12.4 cho GPU RTX)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# 3. Cài đặt các thư viện thực nghiệm NLP & Machine Learning
pip install -r requirements.txt

# 4. Chạy đánh giá VNTC -> Mất ~2.5 phút
python evaluate_vntc.py

# 5. Chạy đánh giá UIT-VSFC -> Mất ~1.2 phút
python evaluate_vsfc.py
```

## 5. Các quyết định thiết kế phương pháp

### 5.1 Kiểm chứng đa thang đo ngữ nghĩa
Việc đánh giá đồng thời VNTC và UIT-VSFC đặt mô hình thử thách trước 2 tác vụ nổi bật của xử lý ngôn ngữ tự nhiên:
- **VNTC (Topic Classification):** Đánh giá khả năng phân cụm từ vựng chuyên ngành trên **văn bản dài**.
- **UIT-VSFC (Sentiment Analysis):** Đánh giá khả năng nhận biết cảm xúc trên **câu ngắn, nhiễu cao**.

### 5.2 Subsampling trên VNTC
Với bộ VNTC gốc nặng (30k Train / 50k Test), thực nghiệm rút gọn mẫu xuống **1.900 dòng Train và 2.500 dòng Test** (giữ đúng tỉ lệ *Test > Train* của bộ gốc), trong khi bộ UIT-VSFC được sử dụng toàn bộ tập Train/Test gốc.

---

## 6. Thiết lập thực nghiệm

| Hạng mục | Bộ dữ liệu 1: VN-News-10 (VNTC) | Bộ dữ liệu 2: UIT-VSFC |
| :--- | :--- | :--- |
| **Tác vụ** | Phân loại chủ đề báo chí (10 lớp) | Phân loại cảm xúc sinh viên (3 lớp: *Neg, Neu, Pos*) |
| **Đặc trưng văn bản** | Văn bản dài (Trung bình ~300–500 từ), chuẩn văn phạm | Câu ngắn (Trung bình ~15–30 từ), ngôn ngữ nói, chứa emoji |
| **Quy mô Train / Test** | 1.900 mẫu / 2.500 mẫu (Subsampled benchmark) | 11.426 mẫu / 3.166 mẫu (Full MTEB benchmark) |
| **Backbone nhúng** | ML-Embed-0.6B | ML-Embed-0.6B
| **Phần cứng** | T4 GPU | T4 GPU
| **Chỉ số đánh giá** | Accuracy (%), Macro F1, Weighted F1 | Accuracy (%), Macro F1, Weighted F1

---

## 7. Kết quả thực nghiệm & Nhận định

**Bảng 1 — Hiệu năng Phân loại 10 chủ đề trên VNTC (1.900 Train / 2.500 Test):**

| Chủ đề | Precision | Recall | F1-Score | Support | Nhận xét|
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Thể thao** | **0.95** | **0.98** | **0.97** | 322 | Từ vựng đặc thù rõ ràng (tên giải đấu, câu lạc bộ, tỷ số) |
| **Vi tính** | **0.88** | **0.95** | **0.92** | 219 | Đặc trưng công nghệ cao, ít bị nhầm lẫn |
| **Thế giới** | **0.92** | **0.89** | **0.91** | 331 | Đặc trưng về từ địa danh, tên riêng quốc tế |
| **Văn hóa** | **0.91** | **0.90** | **0.91** | 323 | Phân loại tốt trên nhóm chủ đề giải trí, nghệ thuật |
| **Kinh doanh**| 0.91 | 0.79 | 0.85 | 266 | Precision cao nhưng có bài bị nhầm sang Chính trị / Xã hội |
| **Sức khỏe** | 0.85 | 0.85 | 0.85 | 254 | Cân bằng giữa Precision và Recall |
| **Pháp luật** | 0.82 | 0.84 | 0.83 | 192 | Đặc trưng với các từ khóa tội phạm, điều tra, tòa án |
| **Chính trị Xã hội** | 0.75 | 0.86 | 0.80 | 368 | Chiếm support lớn nhất, đóng vai trò như lớp trung tâm |
| **Khoa học** | 0.87 | 0.57 | 0.69 | 127 | Recall thấp, nhiều bài bị nhầm sang lớp *Vi tính* hoặc *Sức khỏe* |
| **Đời sống** | 0.57 | 0.60 | **0.58** | 98 | Thiếu mẫu dễ nhầm lẫn với các chủ đề khác |
| **TỔNG THỂ** | **Macro F1:** 0.83 | **Weighted F1:** 0.86 | **Accuracy:** **85.92%** | **2,500** | **Độ chính xác cao** với chỉ 1.900 bài Train |

**Bảng 2 — Hiệu năng Phân loại Cảm xúc trên UIT-VSFC (3.166 Test):**

| Nhãn | Precision | Recall | F1-Score | Support | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Positive** | **0.92** | **0.92** | **0.92** | 1,590 | Phân loại và phát hiện tốt câu dạng khen ngợi |
| **Negative** | **0.88** | **0.94** | **0.91** | 1,409 | Phân loại và phát hiện tốt câu dạng phàn nàn |
| **Neutral** | 0.59 | 0.21 | **0.31** | 167 | Precision và Recall bị ảnh hưởng bởi số lượng câu quá ít |
| **TỔNG THỂ** | **Macro F1:** 0.71 | **Weighted F1:** 0.88 | **Accuracy:** **89.36%** | **3,166** | **Accuracy vượt mốc 89%** nhờ độ chính xác cao ở 2 lớp đa số |

---

## 8. Nhận xét sau thực nghiệm

### 8.1 Vì sao Accuracy trên UIT-VSFC cao (89.36%) nhưng Macro F1 lại thấp (0.71)?
Kết quả thực nghiệm thể hiện một hiện tượng kinh điển trong học máy: **Sự mất cân bằng phân phối lớp gây nhiễu chỉ số Macro**.
- Trong 3,166 mẫu Test của UIT-VSFC, hai lớp `Positive` (1.590 mẫu) và `Negative` (1.409 mẫu) chiếm đến **94.7%** tổng dữ liệu và đều đạt F1 > 0.91. Điều này kéo **Accuracy tổng thể lên mức rất cao (89.36%)** và **Weighted F1 đạt 0.88**.
- Ngược lại, lớp `Neutral` chỉ chiếm **5.3%** (167 mẫu). Khi huấn luyện `LogisticRegression`, đường ranh giới quyết định bị ép nghiêng về 2 lớp đa số, khiến **79% số câu Neutral bị dự đoán nhầm sang Positive hoặc Negative (Recall = 0.21)**.

### 8.2 Nhầm lẫn chủ đề trên VNTC (*Đời sống* và *Khoa học*)
Mặc dù mô hình đạt độ chính xác chung **85.92%** trên tập VNTC rút gọn, kết quả từng lớp chỉ ra 2 cụm có F1-score thấp hơn mức trung bình:
1. **Lớp *Đời sống* (F1 = 0.58):** Là lớp có số mẫu test nhỏ nhất (`support=98`), đồng thời nội dung "đời sống" thường xuyên chia sẻ từ vựng với *Chính trị Xã hội* (tin dân sinh), *Văn hóa* (lối sống) và *Sức khỏe* (sinh hoạt).
2. **Lớp *Khoa học* (Recall = 0.57):** Precision đạt rất cao (0.87) chứng tỏ khi mô hình đã đoán là *Khoa học* thì độ tin cậy rất lớn. Tuy nhiên, Recall thấp chỉ ra rằng gần một nửa số bài báo khoa học đã bị hút vào không gian vector của lớp *Vi tính* (công nghệ ứng dụng) hoặc *Sức khỏe* (y học - sinh học).

---

## 9. Kết luận

Thực nghiệm **Dual-Probing** trên hai bộ dữ liệu chuẩn mực của tiếng Việt (VNTC và UIT-VSFC) đã kiểm chứng thành công năng lực biểu diễn vector vượt trội của **`ML-Embed-0.6B`**. Thay vì fine-tuning tốn kém, việc chỉ cần dùng vector embedding qua ML-Embed-0.6B + Logistic Regression mà vẫn đạt độ chính xác **> 85% – 89%** trên cả hai bài toán đã khẳng định: **Latent space của ML-Embed-0.6B đã tự động ánh xạ và phân cụm tiếng Việt ở độ hoàn thiện cao.**. Không bị ảnh hưởng bởi độ dài của văn bản và tuy có kích thước gọn (~ 600M tham số) nhưng vẫn cho ra kết quả tốt, không đòi hỏi quá nhiều về phần cứng. 

## Tài liệu tham khảo
- [Paper](https://arxiv.org/abs/2605.15081)
- [Source VSFC Dataset](https://www.kaggle.com/datasets/lenhat543/uit-vsfc/data)
- [Source VNTC Dataset](https://www.kaggle.com/datasets/trankhanhquynh/vntc-10topics-1-1/data)
- [Source Model](https://huggingface.co/codefuse-ai/ML-Embed-0.6B)
