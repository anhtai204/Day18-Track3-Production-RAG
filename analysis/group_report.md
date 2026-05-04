# Group Report — Lab 18: Production RAG

**Nhóm:** Nguyễn Anh Tài - Nguyễn Công Quốc Huy (Bàn A2)
**Ngày:** 04/05/2026

## Thành viên & Phân công

| Tên            | Module                          | Hoàn thành | Tests pass |
| -------------- | ------------------------------- | ---------- | ---------- |
| Nguyễn Anh Tài | M1: Chunking, M2: Hybrid Search | ☑          | 5/5        |
| [Thành viên 2] | M3: Reranking                   | ☑          | 5/5        |
| [Thành viên 3] | M4: Evaluation                  | ☑          | 4/4        |

## Kết quả RAGAS

| Metric            | Naive  | Production | Δ       |
| ----------------- | ------ | ---------- | ------- |
| Faithfulness      | 0.9500 | 1.0000     | +0.0500 |
| Answer Relevancy  | 0.7632 | 0.8207     | +0.0575 |
| Context Precision | 0.9333 | 1.0000     | +0.0667 |
| Context Recall    | 0.8000 | 1.0000     | +0.2000 |

## Key Findings

1. **Biggest improvement:** Chiến thuật **"Search on Child, Retrieve Parent"** giúp tăng chỉ số Context Recall lên mức tuyệt đối (1.0). Việc tìm kiếm trên các đoạn nhỏ giúp tăng độ chính xác, trong khi trả về đoạn cha giúp LLM có đầy đủ ngữ cảnh để trả lời.
2. **Biggest challenge:** Xử lý sự không đồng nhất của Tokenizer tiếng Việt trong BM25 (từ ghép `_` vs từ đơn). Việc thiết kế lại hàm tách từ để hỗ trợ cả hai dạng đã giúp hệ thống không còn bị bỏ sót từ khóa quan trọng.
3. **Surprise finding:** Một hệ thống đơn giản có thể đạt điểm Faithfulness rất cao (0.95) nếu dữ liệu sạch, nhưng chỉ có các hệ thống Production thực thụ mới có khả năng bao phủ (Recall) toàn bộ các chi tiết lắt léo trong văn bản pháp luật.

## Presentation Notes (5 phút)

1. **RAGAS scores:** Nhấn mạnh vào việc đạt điểm 1.0 ở 3/4 metric quan trọng nhất.
2. **Biggest win:** Module M2 (Hybrid Search) kết hợp với kỹ thuật mở rộng ngữ cảnh Cha-Con. Đây là phần cốt lõi giúp Production vượt mặt Naive Baseline.
3. **Case study:** Phân tích câu hỏi về "Dữ liệu cá nhân nhạy cảm". Baseline có thể lấy trúng 1 đoạn, nhưng Production lấy được toàn bộ điều luật liên quan, giúp câu trả lời đầy đủ và chuyên nghiệp hơn.
4. **Next optimization:** Tích hợp M5 (Enrichment) hoàn chỉnh để tự động trích xuất Metadata, giúp tìm kiếm theo thuộc tính (Filtering) thay vì chỉ tìm kiếm nội dung.
