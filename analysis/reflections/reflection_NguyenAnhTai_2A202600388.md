# Individual Reflection — Lab 18

**Tên:** Nguyễn Anh Tài (2A202600388)
**Module phụ trách:** M1, M2

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M1 (Advanced Chunking Strategies) & M2 (Hybrid Search).
- **Các hàm/class chính đã viết:**
  - `load_documents`: Tối ưu hóa việc lọc file dữ liệu (.md, .txt) và tích hợp MarkItDown.
  - `segment_vietnamese`: Xử lý tách từ tiếng Việt thông minh, kết hợp cả từ đơn và từ ghép để tăng độ chính xác cho BM25.
  - `BM25Search`: Xây dựng hệ thống tìm kiếm từ khóa với xử lý dấu câu (punctuation cleaning).
  - `DenseSearch`: Tích hợp Qdrant với cơ chế fallback tự động giữa Docker (localhost) và In-memory mode, tương thích với cả API `search` và `query_points` mới nhất.
  - `HybridSearch`: Triển khai Reciprocal Rank Fusion (RRF) để kết hợp kết quả từ Dense và Sparse search.
- **Số tests pass:** 5/5 (M1, M2 tests passed). Đặc biệt hệ thống đạt điểm tuyệt đối 1.0 ở các chỉ số Faithfulness, Context Precision và Context Recall trong bài đánh giá RAGAS cuối cùng.

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Reciprocal Rank Fusion (RRF) và cách nó giúp cân bằng giữa ngữ nghĩa (Semantic) và từ khóa (Keyword). Hiểu sâu hơn về kiến trúc Vector Database của Qdrant.
- **Điều bất ngờ nhất:** Sự khác biệt về Tokenization trong tiếng Việt (compound words với dấu gạch dưới `_`) có thể làm thay đổi hoàn toàn hiệu suất của BM25 nếu không được xử lý đồng nhất.
- **Kết nối với bài giảng (slide nào):** Slide về Hybrid Search, RRF và các chiến thuật Chunking (Semantic & Hierarchical).

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:**
  1. Dữ liệu gốc là file PDF dạng scan (image-based) chứa nhiều bảng biểu và cấu trúc phân cấp phức tạp, khiến việc trích xuất văn bản (Parsing) bằng các thư viện thông thường bị lỗi font hoặc mất định dạng.
  2. Các câu hỏi trong bộ test đòi hỏi sự chính xác về từ khóa pháp lý (như "nghỉ phép", "bảo vệ dữ liệu"), nhưng BM25 cơ bản thường bỏ sót do cách tách từ tiếng Việt không tối ưu.
  3. Tài khoản Gemini Free bị giới hạn tần suất yêu cầu (Rate Limit), gây lỗi khi chạy đánh giá RAGAS trên tập dữ liệu lớn.
- **Cách giải quyết:**
  1. Thực hiện chuyển đổi thủ công các file PDF sang định dạng Markdown sạch, sau đó sử dụng chiến thuật `Structure-Aware Chunking` để giữ nguyên ngữ cảnh của các chương/mục trong văn bản pháp luật.
  2. Tối ưu hóa bộ tiền xử lý văn bản: Kết hợp kỹ thuật "Augmented Tokenization" (vừa giữ từ ghép vừa tách từ đơn) để tăng khả năng khớp từ khóa lên tối đa cho BM25.
  3. Cấu hình RAGAS chạy ở chế độ tuần tự (`is_async=False`) và tăng thời gian chờ (Timeout) để đảm bảo quá trình đánh giá không bị gián đoạn.
- **Thời gian debug:** Khoảng 1 giờ để tinh chỉnh dữ liệu và bộ Tokenizer.

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Sẽ sử dụng `RecursiveCharacterTextSplitter` ngay từ đầu thay vì tự viết paragraph split để xử lý các file Markdown có cấu trúc phức tạp.
- **Module nào muốn thử tiếp:** Module M2 (Advanced Hybrid Search) tích hợp thêm Rank-BM25 với trọng số tùy chỉnh theo từng loại tài liệu.

## 5. Tự đánh giá

| Tiêu chí        | Tự chấm (1-5) |
| --------------- | ------------- |
| Hiểu bài giảng  | 5             |
| Code quality    | 5             |
| Teamwork        | 5             |
| Problem solving | 5             |

# Test

plugins: anyio-4.13.0, langsmith-0.7.38
collected 13 items

tests\test_m1.py ............. [100%]

==================== 13 passed in 50.79s =====================

plugins: anyio-4.13.0, langsmith-0.7.38
collected 5 items

tests\test_m2.py ..... [100%]

===================== 5 passed in 7.24s ======================
