# Bài viết ngắn — Lab 25: GPU FinOps Optimization (NimbusAI)

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Tổng chi phí GPU/tháng | $27,133 | $14,626 | **$12,507 (46%)** |
| Inference | $6.488/1M-token | $1.126/1M-token | 82.6% |
| Purchasing (compute) | $25,667/tháng | $15,627/tháng | 39.1% |

Tổng savings 46% nằm giữa khoảng mục tiêu 40–95% mà `verify.py` kiểm tra, và được ghép từ 4 đòn bẩy độc lập (xem mục 2). Điểm mấu chốt: đo bằng `$/1M-token` (không phải `$/GPU-giờ`) cho thấy hiệu quả sử dụng thực sự — cùng một khoản chi GPU, nếu tối ưu tốt phục vụ được nhiều token hơn hẳn.

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | Tiết kiệm/tháng | % trên tổng |
|---|---|---|
| Purchasing (spot/reserved) | $10,040 | 80.3% |
| Inference (cascade/cache/batch) | $1,212 | 9.7% |
| Right-size util-lies | $655 | 5.2% |
| Kill idle GPUs | $600 | 4.8% |

**Purchasing đóng góp lớn nhất (80%)** vì đây là dạng chi phí có đòn bẩy cao nhất trong dataset: các job training 24/7 hoặc gần-24/7 (`job-train-llm`, `job-infer-chat`...) chiếm phần lớn GPU-giờ, và chuyển đúng sang spot (job có thể gián đoạn) hoặc reserved (job duty cycle cao) cắt trực tiếp giá theo giờ 30–55%.

**Inference chỉ đóng góp ~10%** dù bản thân đòn bẩy này giảm `$/1M-token` tới 82.6% — vì tổng chi phí inference/ngày ($48.87 → $8.48) nhỏ hơn nhiều so với hoá đơn compute hàng tháng. Đây là insight quan trọng: **đòn bẩy có % giảm ấn tượng nhất không nhất thiết là đòn bẩy tiết kiệm được nhiều tiền nhất** — phải nhân với quy mô chi tiêu gốc.

**Right-size + Kill idle** tuy nhỏ (10% gộp lại) nhưng gần như miễn phí để triển khai (không đổi hardware, không risk gián đoạn) — nên xếp ưu tiên triển khai *trước* (quick win) dù giá trị tuyệt đối thấp hơn.

## 3. GPU-Util Lie

`gpu-h100-4` đạt **GPU-Util 98.2%** nhưng **MFU chỉ 0.194** (19.4% FLOPs đỉnh). `gpu-a10g-1` tương tự: **Util 96.9%, MFU 0.268**.

**Tại sao Util cao mà MFU thấp?** `nvidia-smi`/GPU-Util đo "SM có đang nhận lệnh hay không" trong mỗi tick lấy mẫu — một GPU đang chờ dữ liệu từ HBM (memory stall), chờ kernel launch, hoặc chạy kernel có occupancy thấp vẫn được tính là "busy". GPU không hề rảnh về mặt đồng hồ, nhưng phần lớn thời gian "bận" đó là chờ đợi, không phải tính toán hữu ích. Với LLM training/serving, nguyên nhân phổ biến: batch size nhỏ, communication overhead giữa các GPU (all-reduce), hoặc workload bị memory-bound trong khi ta kỳ vọng nó compute-bound.

**Tác động tài chính:** ta đang trả **100% giá thuê H100 on-demand ($2.50/giờ)** nhưng chỉ nhận được **~1/5 số FLOPs** mà con số đó đại diện. Đây là 2 khoản tiết kiệm khác nguồn, không được cộng dồn hai lần trong báo cáo:
- Right-sizing **đúng 2 GPU bị gắn cờ "lie"** (`gpu-h100-4`: H100→A100; `gpu-a10g-1`: A10G→L4, downgrade một bậc theo `RIGHTSIZE_MAP`) tiết kiệm **$655/tháng** — đây là lever "Right-size util-lies" trong bảng M5.
- Riêng biệt, Extension 2 (góc nhìn memory-bound theo MBU, không phải tiêu chí "lie" MFU<30%) quét toàn bộ fleet và phát hiện **6/11 GPU** (toàn bộ nhóm H100 — kể cả những GPU MFU đã tốt như `gpu-h100-0`..`gpu-h100-3`) phần lớn thời gian chạy ở chế độ memory-bound; GPU thay thế khả thi duy nhất vừa rẻ hơn vừa đủ băng thông là **MI300X** (bandwidth cao hơn 158%, giá/GB-VRAM rẻ hơn 3×) → tiết kiệm thêm **$2,376/tháng**. (9/11 GPU trong fleet thực ra thiên về memory-bound, nhưng chỉ nhóm H100 có GPU thay thế rẻ hơn đáp ứng đủ băng thông — A10G/A100/L4 đã ở gần đáy bảng giá nên không tìm được lựa chọn rẻ hơn.)

## 4. Phần mở rộng đã làm (cả 5/5)

### Extension 1 — `recommend_tier()` v2
Thêm (a) interruption-rate riêng theo GPU family (H100 spot ~2%/giờ vs A10G ~8%/giờ) và (b) so sánh reserved 1yr vs 3yr theo bản chất job (`kind=="infer"` → coi như evergreen, dùng 3yr; job train/dev → dùng đúng `days` quan sát được). Kết quả: `job-dev-sandbox` (A10G, interruptible) bị đẩy khỏi spot vì rủi ro gián đoạn 8%/giờ vượt ngưỡng chấp nhận 5%/giờ → chuyển sang on-demand, tốn thêm $277/tháng nhưng loại bỏ rủi ro rework do bị thu hồi liên tục. **Insight:** chính sách "tối ưu hơn" không phải lúc nào cũng rẻ hơn — nó điều chỉnh theo rủi ro thực tế thay vì áp dụng máy móc "cứ interruptible là spot".

### Extension 2 — Right-sizing theo MBU
Phân loại từng GPU-giờ theo roofline regime (arithmetic intensity = achieved_tflops/achieved_bw so với ridge point của GPU đó) thay vì chỉ nhìn MFU thấp. Kết quả: **9/11 GPU trong fleet** (toàn bộ H100, phần lớn A10G/A100/L4) thực ra thiên về memory-bound đa số thời gian — nhưng chỉ nhóm **6 H100** tìm được GPU thay thế vừa rẻ hơn vừa đủ băng thông (MI300X: giữ được 158% băng thông, $/GB-VRAM giảm từ $0.031 xuống $0.010) → tiết kiệm **$2,376/tháng**; A10G/A100/L4 đã ở gần đáy bảng giá nên không có lựa chọn thay thế rẻ hơn thoả điều kiện băng thông ≥70%. Lý do không chọn GPU rẻ nhất theo `$/GPU-giờ`: A10G/L4 rẻ hơn nhưng băng thông quá thấp, sẽ làm workload memory-bound chạy chậm hơn, tăng wall-clock time và có thể triệt tiêu phần tiết kiệm giá thuê.

### Extension 3 — `cache_is_worth_it()`
Viết hàm tính số lần đọc hoà vốn (breakeven reads) khi cache có phí ghi (write premium ~1.25× giá input thường). Với dataset này: breakeven ≈ 1.39 lần đọc/prefix, trong khi thực tế trung bình 237.75 lần (tier small) và 62.25 lần (tier large) → cache **luôn đáng dùng** ở cả hai tier. Nếu traffic caching thực tế thấp hơn ~1.4 lần đọc/prefix (ví dụ hệ thống mới, ít lặp lại prompt), cache sẽ **lỗ** — công thức này giúp tránh bật cache "mù quáng".

### Extension 4 — Ngân sách Reasoning
Reasoning traffic chỉ chiếm **8.4%** tổng số request (201/2,400) nhưng đã chiếm **16.5%** tổng chi phí $ và **tổng cộng ~15.8×** năng lượng so với traffic thường trong ngày (29,788 Wh/ngày so với 1,888 Wh/ngày, dù nhóm reasoning có ít request hơn 11× về số lượng). Con số 15.8× "tổng nhóm" này khác với hệ số **80× năng lượng/query** dùng trong `sustainability.wh_per_query()` (per-query, giả định cùng số token) — mỗi *query* reasoning riêng lẻ tốn gấp 80 lần một query thường có cùng độ dài, nhưng vì reasoning chỉ là 8.4% lưu lượng nên khi cộng dồn cả ngày, tổng năng lượng reasoning "chỉ" gấp ~15.8× tổng năng lượng traffic thường (2 số không nên so trực tiếp vì khác đơn vị: per-query vs. tổng-nhóm-không-cân-kích-thước). Giới hạn xuống 5% traffic tiết kiệm $0.56/ngày nhưng **12,004 Wh/ngày (40% năng lượng reasoning)** — chênh lệch giữa $ và Wh cho thấy nếu chỉ tối ưu theo chi phí, ta sẽ đánh giá thấp tác động năng lượng/carbon của reasoning.

### Extension 5 — Carbon-aware Scheduling
5 job `interruptible=1` chuyển từ `us-east-1` (380 gCO2/kWh) sang `europe-north1` (30 gCO2/kWh, thuỷ điện Na Uy) giảm **92.1% carbon** mỗi job, tổng **626.1 kg CO2e** trong cửa sổ chạy hiện tại. Vì các job này vốn đã chấp nhận gián đoạn/di dời (để dùng spot), chuyển vùng địa lý gần như không thêm rủi ro vận hành.

## 5. Khuyến nghị cho NimbusAI (3 hành động đầu tiên)

1. **Sửa purchasing tier ngay (0 rủi ro kỹ thuật, ROI cao nhất — $10,040/tháng):** Áp dụng lại chính sách tier v2 cho toàn bộ fleet, đặc biệt review các job đang chạy on-demand 24/7 mà đáng lẽ nên reserved, và audit workload nào đang "được gắn nhãn interruptible" trên GPU có tỷ lệ thu hồi cao — chuyển sang on-demand có kiểm soát thay vì chịu rework liên tục.
2. **Right-size 6 GPU H100 "GPU-Util lie" sang MI300X trong 2 tuần tới ($655–$2,376/tháng, effort thấp):** Đây là quick win không cần đổi kiến trúc, chỉ cần benchmark lại workload memory-bound trên MI300X trước khi chuyển toàn bộ fleet.
3. **Thiết lập routing rule giới hạn reasoning ở ~5% traffic + chuyển toàn bộ job interruptible sang `europe-north1`:** Hai việc này gộp lại vừa cắt giảm rõ rệt carbon footprint (giảm >600 kg CO2e + 40% năng lượng reasoning) vừa là nền tảng để NimbusAI báo cáo ESG cho nhà đầu tư — carbon và cost đi cùng hướng ở đây nên không có trade-off phải đánh đổi.

**Ưu tiên theo ROI/effort:** (1) purchasing > (2) right-sizing > (3) reasoning cap + carbon region — đúng thứ tự impact giảm dần và effort tăng dần khi triển khai thực tế (purchasing chỉ là thay đổi tag mua hàng, right-sizing cần benchmark, thay đổi routing/region cần thay đổi hạ tầng triển khai).
