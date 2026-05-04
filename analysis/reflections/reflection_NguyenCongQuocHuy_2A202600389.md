# Individual Reflection — Lab 18

**Tên:** Nguyễn Công Quốc Huy (2A202600389)
**Module phụ trách:** M3 (Reranking) & M4 (Evaluation)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** 
    - **M3 (Reranking):** Triển khai bộ xếp hạng lại sử dụng Cross-Encoder để tối ưu hóa kết quả tìm kiếm từ 40 đoạn xuống 5 đoạn chất lượng nhất.
    - **M4 (Evaluation):** Xây dựng hệ thống đánh giá tự động sử dụng thư viện RAGAS, hỗ trợ đa mô hình (OpenAI & Gemini).
- **Các hàm/class chính đã viết:**
    - `CrossEncoderReranker`: Sử dụng model `BAAI/bge-reranker-v2-m3` để chấm điểm lại sự liên quan giữa Query và Context.
    - `evaluate_ragas`: Hàm cốt lõi để tính toán các chỉ số Faithfulness, Answer Relevancy, Context Precision/Recall.
    - `to_pandas().mean()`: Kỹ thuật xử lý dữ liệu RAGAS linh hoạt bằng Pandas để tránh các lỗi `AttributeError` khi thư viện cập nhật phiên bản.
- **Số tests pass:** 4/4 (Vượt qua tất cả các bài test đánh giá hiệu năng, đạt điểm tuyệt đối 1.0 ở 3 chỉ số quan trọng nhất).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Hiểu sâu về sự khác biệt giữa **Bi-Encoder** (dùng để tìm kiếm nhanh) và **Cross-Encoder** (dùng để xếp hạng chính xác cao nhưng chậm hơn).
- **Điều bất ngờ nhất:** Việc tăng số lượng context (Top-K) không phải lúc nào cũng tốt; chỉ khi kết hợp với Reranker, hệ thống mới thực sự "thông minh" hơn trong việc chọn lọc thông tin.
- **Kết nối với bài giảng:** Bám sát nội dung Slide về "RAG Evaluation" và "Advanced Retrieval Techniques" (Reranking & Hybrid Search).

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Lỗi quota API khi chạy đánh giá RAGAS quá nhiều lần và cấu trúc dữ liệu trả về của RAGAS không ổn định giữa các phiên bản.
- **Cách giải quyết:** Sử dụng kỹ thuật lập trình "phòng thủ" (defensive programming) bằng cách chuyển đổi kết quả sang Pandas DataFrame để truy xuất thuộc tính an toàn hơn và chuyển đổi linh hoạt giữa OpenAI/Gemini.
- **Thời gian debug:** ~1.5 giờ (tập trung vào việc sửa lỗi truy xuất metric của RAGAS và tối ưu hóa độ trễ của Reranker).

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Sẽ triển khai Reranker dưới dạng một Microservice độc lập để tối ưu hóa bộ nhớ RAM cho ứng dụng chính.
- **Module nào muốn thử tiếp:** Module M5 (Enrichment) để thực hiện "Contextual Prepend" nhằm tăng cường ý nghĩa cho từng chunk dữ liệu.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |
