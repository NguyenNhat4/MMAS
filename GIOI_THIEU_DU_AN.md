# ⚕️ Trợ lý Y tế Đa tác nhân (Multi-Agent Medical Assistant)

Dự án này là một hệ thống **chatbot AI tiên tiến** được thiết kế để hỗ trợ chẩn đoán y tế, nghiên cứu và tương tác với bệnh nhân. Hệ thống sử dụng kiến trúc đa tác nhân (multi-agent) để xử lý các tác vụ phức tạp.

## 📌 Tổng quan

**Multi-Agent Medical Assistant** tích hợp nhiều công nghệ AI hiện đại bao gồm:
- **🤖 Mô hình ngôn ngữ lớn (LLMs):** Để hiểu và tạo văn bản y tế.
- **🖼️ Mô hình thị giác máy tính (Computer Vision):** Để phân tích hình ảnh y tế (MRI, X-quang, v.v.).
- **📚 RAG (Retrieval-Augmented Generation):** Truy xuất thông tin từ cơ sở dữ liệu vector để cung cấp câu trả lời chính xác dựa trên tài liệu.
- **🌐 Tìm kiếm Web thời gian thực:** Cập nhật các nghiên cứu y tế mới nhất.
- **👨‍⚕️ Xác minh bởi con người (Human-in-the-Loop):** Đảm bảo độ chính xác cho các chẩn đoán quan trọng.

## ✨ Các tính năng chính

1.  **Kiến trúc Đa tác nhân (Multi-Agent):** Các tác nhân chuyên biệt làm việc phối hợp để xử lý chẩn đoán, truy xuất thông tin, và suy luận.
2.  **Hệ thống RAG tiên tiến:**
    *   Sử dụng **Docling** để trích xuất văn bản, bảng biểu và hình ảnh từ tài liệu PDF.
    *   Tìm kiếm lai (Hybrid search) sử dụng **Qdrant** (kết hợp từ khóa BM25 và vector embedding).
    *   Sắp xếp lại (Reranking) kết quả tìm kiếm để tăng độ chính xác.
    *   Cung cấp liên kết đến tài liệu nguồn trong câu trả lời.
3.  **Phân tích hình ảnh y tế:**
    *   Phát hiện khối u não (Brain Tumor Detection).
    *   Phân loại bệnh qua X-quang ngực (Chest X-ray Disease Classification).
    *   Phân đoạn tổn thương da (Skin Lesion Segmentation).
4.  **Tích hợp nghiên cứu thời gian thực:** Tác nhân tìm kiếm web giúp truy xuất các bài báo y khoa mới nhất.
5.  **Tương tác giọng nói:** Hỗ trợ chuyển đổi giọng nói thành văn bản (Speech-to-Text) và văn bản thành giọng nói (Text-to-Speech) qua Eleven Labs API.
6.  **Giao diện trực quan:** Dễ dàng sử dụng cho các chuyên gia y tế.

## 🛠️ Công nghệ sử dụng (Tech Stack)

| Thành phần | Công nghệ |
|---|---|
| **Backend** | FastAPI |
| **Điều phối Agent** | LangGraph, LangChain |
| **Cơ sở dữ liệu Vector** | Qdrant |
| **Xử lý tài liệu** | Docling |
| **Thị giác máy tính** | PyTorch (Segmentation, Object Detection, Classification) |
| **Xử lý giọng nói** | Eleven Labs API |
| **Frontend** | HTML, CSS, JavaScript |
| **Triển khai** | Docker |

## 🚀 Cách sử dụng cơ bản

Sau khi cài đặt và chạy ứng dụng (qua Docker hoặc chạy trực tiếp `app.py`), bạn có thể:

1.  **Tải lên hình ảnh y tế:** Sử dụng các tác nhân Vision để chẩn đoán dựa trên hình ảnh (ví dụ: ảnh MRI não, ảnh chụp da).
2.  **Hỏi đáp y tế:** Đặt câu hỏi để hệ thống tìm kiếm trong cơ sở tri thức (RAG) hoặc tìm kiếm trên web.
3.  **Tương tác bằng giọng nói:** Sử dụng tính năng voice để giao tiếp với trợ lý.
4.  **Xác minh:** Các chuyên gia y tế có thể xem xét và xác minh các kết quả do AI đưa ra trước khi xuất kết quả cuối cùng.

---
*Dự án này được xây dựng nhằm mục đích nghiên cứu và hỗ trợ, không thay thế hoàn toàn cho chẩn đoán y khoa chuyên nghiệp.*
