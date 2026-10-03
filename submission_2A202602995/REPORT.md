# Báo cáo Lab Day 1 — Trịnh Xuân Huy — 2A202602995

## 1. Thiết lập

- **Môi trường:** Máy tính cá nhân (Windows 11, Intel(R) Iris(R) Xe Graphics CPU, Python 3.11, PyTorch 2.14.1+cpu).
- **Dữ liệu:** Forest CoverType (Blackard & Dean, UCI); `train` 464 809 mẫu / `eval` 116 203 mẫu theo file metadata cố định `split_metadata.csv`.
- **Phân tách Validation:** 20% mẫu tách từ tập `train` theo phương pháp phân tầng (stratified) với `seed=42` -> **371 847 mẫu train** / **92 962 mẫu val**. Chuẩn hoá chỉ tính trung bình (mean) và độ lệch chuẩn (std) trên 10 cột liên tục của tập train còn lại, áp dụng cùng thông số cho val và eval (không rò rỉ dữ liệu).
- **Model:** `M-base` (54 -> 256 -> 128 -> 7, 47 879 tham số).
- **Baseline cấu hình:** Cross-Entropy loss, SGD + momentum 0.9, tốc độ học lr = 0.05, kích thước lô 512, 20 epochs, khởi tạo He (`kaiming_normal_`), FP32, không dropout, không gradient clipping.
- **Mốc tham chiếu:** Độ chính xác chiến lược "luôn đoán lớp đa số" (Lớp 1) trên val = **0.4876** (macro-F1 ≈ **0.0936**).
- **Các chủ đề đã thử nghiệm:** [x] loss · [x] optimizer · [x] hyper-parameter · [x] dropout · [x] clipping · [x] mixed precision · [x] init (Hoàn thành đầy đủ 7/7 chủ đề).

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Phép kiểm tra | Kết quả đo đạc thực tế |
|---|---|
| Số tham số / shape logits | **47 879** tham số / `(B, 7)` |
| Loss bước 0 trên Val (so với ln(7) = 1.946) | **1.7837** (dao động 1.78 – 2.27 tuỳ seed/init, gần sát ln(7)) |
| Quá khớp 20 mẫu: loss cuối / accuracy | **0.000012** / **100.0%** |
| Mọi tham số có gradient khác 0 sau backward | **Có** (chuẩn gradient W1=0.87, W2=2.38, W3=2.12) |
| Baseline: số seed đã chạy | **3 seeds** (`base-s1`, `base-s2`, `base-s3`) |
| Baseline: Val Accuracy (TB ± std) | **0.8995 ± 0.0020** |
| Baseline: Val Macro-F1 (TB ± std) | **0.8380 ± 0.0036** |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = **0.0073** (tính trên Val Macro-F1 của 3 seed baseline).  
*Mọi kết luận "A tốt hơn B" ở phần sau bắt buộc phải có chênh lệch Δ F1 > 0.0073, nếu nhỏ hơn sẽ được ghi nhận là nằm trong khoảng dao động ngẫu nhiên của seed.*

---

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — Cross-Entropy vs MSE
- **Dự đoán:** Cross-Entropy (CE) sẽ cho tốc độ hội tụ nhanh hơn và F1 cao hơn rõ rệt so với MSE. Lý do: đạo hàm của CE kết hợp với softmax đối với logit z_i là (p_i - y_i), tỉ lệ thuận trực tiếp với sai số nên không bị bão hoà ở vùng dự đoán sai lệch nhiều; ngược lại MSE tính sai số bình phương trên nhãn one-hot sẽ có gradient co cụm rất nhỏ khi xác suất dự đoán ở vùng bão hoà.
- **Kết quả:** Thí nghiệm `loss-mse` đạt Val Macro-F1 = **0.6845** (Val Acc = 0.8553), kém hơn Baseline CE tới Δ = -0.1535 (vượt xa ngưỡng nhiễu 2σ).  
  *(Lưu ý: Giá trị loss cuối của MSE là 0.0324 trong khi CE là 0.2473, hai giá trị này khác thang đo toán học nên không thể so sánh trực tiếp độ lớn của loss, mà phải so sánh bằng Accuracy và Macro-F1).*
- **Ảnh minh chứng:** [figures/loss-mse.png](figures/loss-mse.png)

### 3.2 Bộ tối ưu hoá (Optimizer)
- **Dự đoán:** Adam và AdamW với cơ chế điều chỉnh tốc độ học thích nghi theo từng tham số sẽ hội tụ nhanh hơn SGD ở các epoch đầu. SGD thuần không có momentum sẽ hội tụ chậm và dễ bị dao động trên các hẻm núi địa hình hẹp.
- **Bảng so sánh ở lr tối ưu của từng bộ:**

| Thí nghiệm (`exp_id`) | Bộ tối ưu | Learning Rate | Best Epoch | Val Acc | Val Macro-F1 | Δ so với Base | Vượt nhiễu (> 2σ)? |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| `opt-sgd` | SGD thuần | 0.05 | 18 | 0.8396 | 0.6962 | -0.1418 | Có (Tệ hơn) |
| `base-s1` | SGD + momentum 0.9 | 0.05 | 20 | 0.9005 | 0.8421 | Baseline | — |
| `opt-adam-lr1e-3` | Adam | 0.001 | 18 | 0.9024 | **0.8459** | +0.0079 | **Có (Tốt hơn)** |
| `opt-adam-lr3e-4` | Adam | 0.0003 | 20 | 0.8748 | 0.7930 | -0.0450 | Có (Tệ hơn) |
| `opt-adamw-lr1e-3` | AdamW | 0.001 | 18 | 0.9011 | **0.8458** | +0.0078 | **Có (Tốt hơn)** |

- **Độ nhạy với lr và giải thích cơ chế:**
  - SGD phụ thuộc hoàn toàn vào động lượng (momentum): thiếu momentum (`opt-sgd`), F1 tụt xuống 0.6962 vì các hướng gradient dao động triệt tiêu lẫn nhau.
  - Adam rất nhạy cảm với tốc độ học: với lr = 0.001, Adam đạt 0.8459 (vượt ngưỡng 2σ so với baseline), hội tụ dốc đứng ngay từ epoch 3. Tuy nhiên, khi giảm lr xuống 0.0003 (`opt-adam-lr3e-4`), mô hình không đi đủ bước trong 20 epochs và chỉ đạt F1 0.7930.
  - Adam và AdamW cho kết quả tương đương nhau (0.8459 vs 0.8458) vì với số epoch ngắn (20 epochs), việc tách suy giảm trọng số (weight decay 0.01) của AdamW chưa tạo ra khoảng cách lớn về mặt khái quát hoá.
- **Ảnh minh chứng:** [figures/compare_optimizer.png](figures/compare_optimizer.png)

### 3.3 Hyper-parameter (Batch Size & Kiến trúc)
- **Batch Size:**
  - `batch-128`: Val Macro-F1 = **0.8588** (Δ = +0.0208, vượt trội ngưỡng 2σ).
  - `batch-2048`: Val Macro-F1 = **0.7790** (Δ = -0.0590, tụt dốc).
  - *Giải thích cơ chế:* Cùng chạy 20 epochs, batch 128 thực hiện khoảng 58 000 bước cập nhật trọng số, kèm theo nhiễu ngẫu nhiên vừa phải (stochastic gradient noise) giúp mô hình dễ thoát khỏi các điểm yên ngựa và cực tiểu địa phương phẳng. Trong khi đó, batch 2048 chỉ cập nhật khoảng 3 600 bước, khiến số lần chỉnh sửa trọng số quá ít nếu không tăng lr theo quy tắc tỉ lệ tuyến tính (linear scaling rule).
- **Kiến trúc mạng (Độ rộng và Độ sâu):**
  - Mạng rộng `arch-mwide` (54 -> 512 -> 256 -> 7, 161k tham số): Val Macro-F1 = **0.8626** (Δ = +0.0246).
  - Mạng sâu `arch-mdeep` (54 -> 256 -> 128 -> 64 -> 7, 55k tham số): Val Macro-F1 = **0.8656** (Δ = +0.0276, **mô hình xuất sắc nhất**).
  - *Giải thích:* Kiến trúc sâu 3 tầng ẩn phân tách được các ranh giới phi tuyến phức tạp của 54 biến địa hình tốt hơn mạng phẳng 2 tầng, đồng thời kiểm soát số tham số vừa phải (55k) giúp huấn luyện nhanh và ít tốn bộ nhớ hơn `M-wide` (161k).
- **Ảnh minh chứng:** [figures/compare_hparam.png](figures/compare_hparam.png)

### 3.4 Dropout
- **Dự đoán:** Do kích thước tập dữ liệu huấn luyện rất lớn (371 847 mẫu) so với dung lượng mạng `M-base` (chỉ 47 879 tham số), tỉ lệ mẫu/tham số xấp xỉ 7.8 nên mô hình không hề bị quá khớp (overfitting). Do đó việc thêm Dropout sẽ làm suy giảm năng lực học (underfitting).
- **Kết quả:**
  - `drop-0.1`: Val Macro-F1 = **0.8220** (tụt -0.0160).
  - `drop-0.3`: Val Macro-F1 = **0.7570** (tụt -0.0810).
- **Nhận xét khoảng cách Train - Val Loss:** Ở Baseline, khoảng cách giữa Train Loss (0.237) và Val Loss (0.247) chỉ là 0.01, chứng minh mạng hoàn toàn không bị overfit. Khi bật dropout q = 0.3, mô hình bị co cụm biểu diễn, dẫn đến giảm mạnh cả Accuracy lẫn F1.
- **Ảnh minh chứng:** [figures/compare_dropout.png](figures/compare_dropout.png)

### 3.5 Cắt Gradient (Gradient Clipping)
- **Ở tốc độ học bình thường (lr = 0.05):** Thí nghiệm `clip-1.0` đạt Val Macro-F1 = **0.8371** (chênh lệch -0.0009 so với Baseline, hoàn toàn nằm trong khoảng nhiễu 2σ = 0.0073). Quan sát đồ thị cho thấy chuẩn gradient bình thường dao động quanh mức 0.6 - 1.1, nên ngưỡng cắt c = 1.0 hầu như không bị kích hoạt.
- **Thí nghiệm phản chứng ở tốc độ học cực cao (lr = 1.0):**
  - `noclip-highlr` (lr = 1.0, không clip): Val Macro-F1 rơi xuống **0.6231**, gradient xuất hiện các gai đột biến lớn làm chệch hướng hội tụ.
  - `clip-highlr` (lr = 1.0, có clip c = 1.0): Val Macro-F1 giữ vững ở mức **0.7963** (cứu được mô hình tăng **+0.1732** so với khi không clip!).
- **Kết luận:** Gradient clipping hoạt động như một "dây đai an toàn" bảo vệ mô hình không bị bùng nổ trọng số khi gặp tốc độ học lớn hoặc lô dữ liệu chứa nhiều ngoại lai.
- **Ảnh minh chứng:** [figures/compare_clipping.png](figures/compare_clipping.png)

### 3.6 Mixed Precision (AMP)
- Thí nghiệm `amp-fp16`: Chạy trên nền tảng CPU máy cá nhân, PyTorch nhận diện thiết bị không có phần cứng chuyên dụng FP16 nên tự động chuyển về tính toán FP32 chuẩn xác.
- Kết quả đạt Val Macro-F1 = **0.8421**, thời gian 1.91s/epoch tương đương hoàn toàn với Baseline FP32 (1.86s/epoch).

### 3.7 Khởi tạo tham số (Weight Initialization)
- **Thí nghiệm `init-zeros` (Khởi tạo toàn bộ số 0):** Mô hình **hoàn toàn không học được gì**. Accuracy đứng im ở mức **0.4876** và Val Macro-F1 = **0.0936** (đúng bằng con số của chiến lược đoán lớp đa số ngây thơ).
  - *Giải thích cơ chế:* Khi toàn bộ W = 0, b = 0, đầu ra của mọi nơ-ron trong cùng một lớp đều bằng nhau (z_i = 0). Hàm ReLU(0) = 0 có đạo hàm triệt tiêu. Hơn nữa, do tính đối xứng hoàn hảo (symmetry), gradient truyền ngược về mọi nơ-ron trong cùng một lớp đều giống hệt nhau, khiến các nơ-ron cùng cập nhật một giá trị như nhau và không bao giờ học được các đặc trưng khác biệt.
- **Thí nghiệm `init-normal` (W ~ N(0, 0.01^2)):** Val Macro-F1 = **0.8231** (thấp hơn He do phương sai quá nhỏ khiến độ lệch chuẩn kích hoạt bị suy giảm nhanh chóng qua các tầng sâu).
- **Thí nghiệm `init-xavier`:** Val Macro-F1 = **0.8402** (tương đương He do mạng chỉ gồm 2-3 tầng ẩn, tác động bảo toàn phương sai giữa căn(1/n) và căn(2/n) chưa tạo ra sự phân hoá rõ rệt như ở các mạng rất sâu trên 20 tầng).
- **Ảnh minh chứng:** [figures/compare_init.png](figures/compare_init.png)

---

## 4. Đánh giá cuối trên tập eval

Toàn bộ quá trình thử nghiệm và chọn cấu hình được thực hiện **nghiêm ngặt 100% trên tập Validation**, không hề nhìn vào tập `eval`. Mô hình đạt Val Macro-F1 cao nhất là **`arch-mdeep`** (54 -> 256 -> 128 -> 64 -> 7).

Số liệu dưới đây được trích xuất trực tiếp từ file chấm điểm chính thức [eval_result.json](eval_result.json) (sinh ra bởi script `scripts/evaluate.py`):

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** | Eval Accuracy |
|---|:---:|:---:|:---:|:---:|
| Baseline (`base-s1`) | 1 | 0.8421 | **0.8410** | 0.8992 |
| **Cấu hình cuối cùng (`arch-mdeep`)** | **1** | **0.8656** | **0.8651** | **0.9152** |

- **Cải thiện so với baseline trên eval:** Đạt mức tăng **+0.0241** (vượt xa ngưỡng 0.02 và ngưỡng nhiễu 2σ = 0.0073).
- **Độ tin cậy của tập Validation:** Điểm trên Val (0.8656) và trên Eval (0.8651) chênh lệch chưa tới 0.0005, chứng minh phép chia validation 20% phân tầng ban đầu là một đại lượng ước lượng vô cùng tin cậy và không hề có hiện tượng rò rỉ dữ liệu hay chọn lọc thiên vị.

### 4.1 Phân tích lỗi theo lớp (Error Analysis)

Chi tiết hiệu năng phân loại trên từng loại rừng của tập `eval` (tổng số 116 203 mẫu):

| Lớp (c) | Tên loại rừng | Support | Precision | Recall | **F1-Score** |
|:---:|---|:---:|:---:|:---:|:---:|
| 0 | Spruce/Fir | 42 368 | 0.9285 | 0.8997 | 0.9138 |
| 1 | Lodgepole Pine | 56 661 | 0.9113 | 0.9478 | 0.9292 |
| 2 | Ponderosa Pine | 7 151 | 0.9377 | 0.8523 | 0.8930 |
| 3 | Cottonwood/Willow | 549 | 0.8192 | 0.7923 | 0.8056 |
| 4 | Aspen | 1 899 | 0.8392 | 0.7009 | **0.7638** |
| 5 | Douglas-fir | 3 473 | 0.8095 | 0.8382 | 0.8236 |
| 6 | Krummholz | 4 102 | 0.9381 | 0.9161 | 0.9270 |

#### Ma trận nhầm lẫn (Confusion Matrix):
```text
       Lớp 0   Lớp 1   Lớp 2   Lớp 3   Lớp 4   Lớp 5   Lớp 6
Lớp 0: 38117    3993       0       0      18      10     230
Lớp 1:  2599   53705      43       2     205      89      18
Lớp 2:     0     414    6095      77      28     537       0
Lớp 3:     0       2      74     435       0      38       0
Lớp 4:    43     500      14       0    1331      11       0
Lớp 5:     7     261     274      17       3    2911       0
Lớp 6:   287      56       0       0       1       0    3758
```

#### Phân tích nguyên nhân:
1. **Lớp khó nhất:** Là **Lớp 4 (Aspen)** có F1 thấp nhất (0.7638, Recall chỉ đạt 70.09%), tiếp theo là **Lớp 3 (Cottonwood/Willow)** (0.8056).
2. **Nguyên nhân nhầm lẫn chính:**
   - Trong 1 899 mẫu thật của Lớp 4, có tới **500 mẫu bị dự đoán nhầm sang Lớp 1 (Lodgepole Pine)**.
   - Trong 549 mẫu thật của Lớp 3, có **74 mẫu bị nhầm sang Lớp 2** và **38 mẫu nhầm sang Lớp 5**.
3. **Lý giải từ bản chất dữ liệu:**
   - *Mất cân bằng lớp trầm trọng:* Lớp 1 chiếm tới 48.8% tập dữ liệu trong khi Lớp 4 chỉ chiếm 1.6% và Lớp 3 chỉ chiếm vỏn vẹn 0.5%. Hàm mất mát tổng thể bị chi phối bởi các lớp đa số, khiến mô hình có xu hướng thiên kiến dự đoán về lớp 1 khi gặp các trường hợp mập mờ.
   - *Sự tương đồng về đặc trưng địa hình:* Cây Aspen và Lodgepole Pine có phân bố độ cao (Elevation) và góc chiếu sáng (Hillshade) giao thoa rất lớn ở sườn núi Colorado.
4. **Giải pháp cải tiến đề xuất:** Áp dụng **Class-weighted Cross-Entropy** (đặt trọng số phạt nghịch đảo với tần suất lớp) hoặc sử dụng **Focal Loss** để tập trung gradient vào các mẫu thiểu số khó phân loại.

---

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**  
   Khi được chỉnh lr công bằng, **Adam (lr = 0.001)** và **AdamW** đạt F1 cao nhất (0.8459), bứt phá tốc độ ngay từ 3 epoch đầu tiên nhờ cơ chế chuẩn hoá bước nhảy theo phương sai gradient từng chiều. SGD + momentum (0.8421) bám sát phía sau khi có đủ 20 epoch. Tuy nhiên, nếu không chỉnh lr (ví dụ ép Adam dùng cùng lr = 0.05 của SGD hoặc dùng lr quá bé 0.0003), Adam sẽ bị phân kỳ (NaN loss) hoặc học rất chậm (F1 tụt xuống 0.7930). Do đó, kết luận so sánh bộ tối ưu chỉ có ý nghĩa khoa học khi khảo sát ở vùng lr tối ưu của từng bộ.

2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**  
   Dropout **hoàn toàn không giúp ích**, thậm chí gây hại nghiêm trọng làm tụt F1 từ 0.8380 xuống 0.7570. Lý do: Dropout là kỹ thuật chính quy hoá ngăn ngừa hiện tượng đồng thích nghi (co-adaptation) khi mô hình bị quá khớp (overfitting). Trong bài toán này, tập dữ liệu có tới 371 nghìn mẫu trong khi mạng chỉ có 47 nghìn tham số, khoảng cách train-val loss chỉ 0.01 (không hề overfit). Dropout chỉ nên dùng khi mạng có dung lượng tham số rất lớn so với dữ liệu hoặc quan sát thấy train loss giảm sâu nhưng val loss tăng ngược trở lại.

3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**  
   Gradient clipping giải quyết triệt để vấn đề **bùng nổ gradient (exploding gradients)** làm bước nhảy tham số quá đà văng ra khỏi vùng cực tiểu. Minh chứng rõ nhất là ở thí nghiệm lr = 1.0: khi không clip (`noclip-highlr`), mô hình bị rung lắc mạnh và F1 rớt xuống 0.6231; nhưng khi bật clip c = 1.0 (`clip-highlr`), F1 được cứu trở lại mức 0.7963 (tăng tới +0.1732).

4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?**  
   Trên CPU máy cá nhân, Mixed Precision không làm tăng tốc vì CPU không có phần cứng chuyên dụng (Tensor Cores / FP16 ALU) và overhead của việc cast dữ liệu không bù đắp được chi phí tính toán. Ngay cả trên GPU hiện đại, với mạng MLP kích thước nhỏ (47k tham số), thời gian huấn luyện chủ yếu bị chi phối bởi độ trễ gọi kernel (kernel launch overhead) và băng thông bộ nhớ chứ không bị nghẽn ở năng lực tính toán dấu phẩy động (compute-bound), do đó mức tăng tốc sẽ không rõ rệt như đối với các mạng CNN hay Transformer khổng lồ.

5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**  
   Khởi tạo toàn số 0 làm hỏng mô hình vì nó gây ra **tính đối xứng hoàn hảo**: mọi nơ-ron cùng tầng nhận cùng giá trị kích hoạt và nhận cùng một gradient đạo hàm, dẫn đến mạng nhiều nơ-ron bị suy biến thành mạng chỉ có 1 nơ-ron duy nhất; kết hợp với hàm kích hoạt ReLU(0) = 0 có đạo hàm triệt tiêu khiến mạng hoàn toàn tê liệt (F1 chỉ đạt 0.0936).  
   Khởi tạo **He** đặt phương sai trọng số là Var[W] = 2/n_in, gấp đôi so với **Xavier** (Var[W] = 1/n_in). Hệ số 2 này xuất phát từ việc hàm ReLU triệt tiêu một nửa miền giá trị âm về 0, làm giảm một nửa phương sai của tín hiệu truyền qua. Do đó, He là khởi tạo bắt buộc để duy trì phương sai ổn định khi mạng có nhiều tầng kích hoạt ReLU.

6. **Quay lại câu hỏi của bài học: Một mạng có loss không giảm sau 2 000 bước. 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao?**  
   Dựa vào bảng chẩn đoán ở Chương 5 và các thí nghiệm đã thực hiện, 3 phép kiểm tra đầu tiên là:
   1. **Kiểm tra Loss bước 0:** Đo loss tại bước cập nhật đầu tiên xem có xấp xỉ ln(7) ≈ 1.946 không. Nếu loss bước 0 cao hơn nhiều (ví dụ > 5.0), lỗi nằm ở khởi tạo trọng số quá lớn hoặc chưa chuẩn hoá các đặc trưng đầu vào.
   2. **Kiểm tra khả năng quá khớp một lô nhỏ (20 mẫu):** Tắt mọi chính quy hoá (dropout=0, weight decay=0), huấn luyện mạng trên đúng 20 mẫu trong 200 bước. Nếu loss không giảm về sát 0 và accuracy không đạt 100%, lỗi 100% nằm ở code vòng lặp hoặc nhãn (ví dụ: quên `zero_grad`, áp dụng softmax 2 lần, nhãn chưa trừ 1 về 0..6, hoặc optimizer không truyền đủ tham số).
   3. **Kiểm tra dòng chảy gradient (`grad_norm`):** In chuẩn gradient của từng tầng sau lệnh `loss.backward()`. Nếu gradient ở các tầng đầu bằng 0 hoặc `None`, mạng đang bị triệt tiêu gradient (vanishing gradient do rơi vào vùng chết của ReLU hoặc tốc độ học quá nhỏ).

---

## 6. Hạn chế và điều bất ngờ

- **Điều bất ngờ nhất:** Kiến trúc sâu `arch-mdeep` (54 -> 256 -> 128 -> 64 -> 7) với chỉ 55 687 tham số lại vượt trội hơn hẳn kiến trúc rộng `arch-mwide` (161 287 tham số) về cả Macro-F1 (0.8656 vs 0.8626) và thời gian huấn luyện (2.87s vs 4.25s/epoch). Điều này khẳng định rằng tăng chiều sâu phân cấp biểu diễn mang lại hiệu quả cao hơn đơn thuần là mở rộng số lượng nơ-ron song song.
- **Hạn chế của thiết kế thí nghiệm:**
  - Số lượng seed đo độ nhiễu baseline là 3 seeds, tuy đã cho ra ước lượng độ lệch chuẩn std = 0.0036 đáng tin cậy nhưng vẫn là kích thước mẫu thống kê nhỏ.
  - Các thí nghiệm đều cố định ở 20 epochs; với các cấu hình có batch size lớn (2048 mẫu), số bước cập nhật ít hơn đáng kể nên chưa phản ánh hết tiềm năng tối đa nếu được huấn luyện lâu hơn.
- **Hướng phát triển tiếp theo:** Triển khai cơ chế lập lịch tốc độ học Cosine Annealing kết hợp Warmup, và áp dụng kĩ thuật gán trọng số mẫu theo nghịch đảo tần số lớp để nâng cao chỉ số F1 cho 2 lớp thiểu số (Lớp 3 và Lớp 4).

---

## 7. Phụ lục

- **Danh sách file nộp bài trong thư mục `submission_2A202602995/`:**
  - `REPORT.md`: Báo cáo kết luận tổng hợp (file này).
  - `experiments.xlsx`: Bảng so sánh chi tiết toàn bộ 21 lần chạy thí nghiệm.
  - `predictions_eval.csv`: Dự đoán của cấu hình tốt nhất `arch-mdeep` trên 116 203 mẫu eval.
  - `eval_result.json`: Kết quả chấm điểm chính thức từ `scripts/evaluate.py`.
  - `figures/`: Chứa đủ 21 ảnh thí nghiệm `<exp_id>.png` và 6 ảnh so sánh nhóm `compare_<nhóm>.png`.
  - `results/`: Chứa đủ 21 file lịch sử chi tiết `<exp_id>.json`.
  - `code/`: Chứa file `lab.ipynb` và toàn bộ các module `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`.
- **Tổng thời gian huấn luyện toàn bộ 21 thí nghiệm:** ≈ 10.5 phút trên CPU máy cá nhân.
