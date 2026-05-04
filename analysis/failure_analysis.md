# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Production RAG Team  
**Thành viên:** Nguyễn Anh Tài (2A202600388) · Nguyễn Công Quốc Huy (2A202600389)

---

## RAGAS Scores

| Metric            | Naive Baseline | Production | Δ       |
| ----------------- | -------------- | ---------- | ------- |
| Faithfulness      | 0.9500         | 1.0000     | +0.0500 |
| Answer Relevancy  | 0.7632         | 0.8207     | +0.0575 |
| Context Precision | 0.9333         | 1.0000     | +0.0667 |
| Context Recall    | 0.8000         | 1.0000     | +0.2000 |

## Bottom-5 Failures

### #1

- **Question:** Dữ liệu cá nhân nhạy cảm là gì?
- **Expected:** Dữ liệu cá nhân nhạy cảm là dữ liệu gắn liền với quyền riêng tư... gồm quan điểm chính trị, tôn giáo, tình trạng sức khỏe...
- **Got:** (LLM trả lời cực kỳ đầy đủ, bao gồm cả phân tích hệ quả pháp lý và các biện pháp bảo vệ theo Nghị định 13).
- **Worst metric:** Answer Relevancy (0.771)
- **Error Tree:** Output đúng (nhưng quá chi tiết) → Context đúng (Parent 2048 chars) → Query OK →
- **Root cause:** Mismatch về độ dài (Length mismatch). LLM cung cấp nhiều thông tin hữu ích hơn Ground Truth khiến điểm Relevancy bị kéo xuống.
- **Suggested fix:** Tinh chỉnh Prompt để LLM trả lời ngắn gọn, tập trung vào liệt kê nếu đề bài yêu cầu "là gì".

### #2

- **Question:** Quy định về bảo vệ dữ liệu cá nhân như thế nào?
- **Expected:** Nghị định 13 quy định về bảo vệ dữ liệu cá nhân và trách nhiệm bảo vệ dữ liệu cá nhân của cơ quan, tổ chức, cá nhân có liên quan.
- **Got:** (LLM liệt kê các quy định cụ thể về trách nhiệm, quyền hạn và các hành vi bị nghiêm cấm).
- **Worst metric:** Answer Relevancy (0.807)
- **Error Tree:** Output đúng → Context đúng → Query OK →
- **Root cause:** Câu hỏi quá rộng (Vague query). LLM chọn cách trả lời chi tiết thay vì tóm tắt khái quát như Ground Truth.
- **Suggested fix:** Sử dụng System Prompt yêu cầu tóm tắt (Summarization style).

### #3

- **Question:** Đối tượng áp dụng của Nghị định 13 là ai?
- **Expected:** Nghị định áp dụng đối với cơ quan, tổ chức, cá nhân Việt Nam; cơ quan, tổ chức, cá nhân nước ngoài tại Việt Nam...
- **Got:** (LLM liệt kê chính xác các điểm a, b, c, d của Điều 1).
- **Worst metric:** Answer Relevancy (0.822)
- **Error Tree:** Output đúng → Context đúng → Query OK →
- **Root cause:** Style mismatch. LLM sử dụng định dạng danh sách (bullet points) trong khi Ground Truth là văn bản phẳng.
- **Suggested fix:** Đồng bộ hóa định dạng trả về giữa mẫu và LLM.

### #4

- **Question:** Tên người nộp thuế trong tờ khai BCTC là gì?
- **Expected:** CÔNG TY CỔ PHẦN DHA SURFACES.
- **Got:** Tên người nộp thuế trong tờ khai BCTC là CÔNG TY CỔ PHẦN DHA SURFACES.
- **Worst metric:** Answer Relevancy (0.834)
- **Error Tree:** Output đúng → Context đúng → Query OK →
- **Root cause:** LLM lặp lại câu hỏi trong câu trả lời, làm loãng các từ khóa so với Ground Truth chỉ có tên công ty.
- **Suggested fix:** Prompt yêu cầu chỉ trả về thực thể (Entity extraction mode).

### #5

- **Question:** Dữ liệu cá nhân cơ bản bao gồm những thông tin gì?
- **Expected:** Họ và tên, ngày tháng năm sinh, giới tính, nơi sinh, quốc tịch, số điện thoại...
- **Got:** (LLM liệt kê đầy đủ 12 hạng mục theo Điều 2 Nghị định 13).
- **Worst metric:** Answer Relevancy (0.867)
- **Error Tree:** Output đúng → Context đúng → Query OK →
- **Root cause:** Độ bao phủ thông tin của LLM tốt hơn mẫu.
- **Suggested fix:** Cập nhật Ground Truth chuẩn hóa theo đúng văn bản luật.

## Case Study (cho presentation)

**Question chọn phân tích:** Dữ liệu cá nhân nhạy cảm là gì?

**Error Tree walkthrough:**

1. Output đúng? → ĐÚNG. LLM đã trả lời chính xác và đầy đủ các loại dữ liệu nhạy cảm.
2. Context đúng? → ĐÚNG. Kỹ thuật Parent Retrieval đã lấy được toàn bộ Điều 2 của Nghị định 13.
3. Query rewrite OK? → OK. Hybrid Search tìm được đúng đoạn văn bản chứa từ khóa.
4. Fix ở bước: Bước Generation (Prompt engineering) để khớp văn phong với Ground Truth.

**Nếu có thêm 1 giờ, sẽ optimize:**

- Tích hợp thêm module M5 Enrichment để gắn tag metadata cho từng Điều luật, giúp tăng tốc độ tìm kiếm và độ chính xác của Context Precision.
