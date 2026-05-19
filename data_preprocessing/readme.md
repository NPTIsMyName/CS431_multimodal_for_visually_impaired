1. gen_data.ipynb - Sinh câu trả lời bằng AI
Notebook này chịu trách nhiệm tương tác với API của Pixtral/Mistral để sinh ra các câu trả lời tự động dựa trên hình ảnh và câu hỏi đầu vào.

Đầu vào: Các file JSON chứa metadata và câu hỏi từ tập dữ liệu VizWiz (ví dụ: metadata_ans_0.json và metadata_ans_1.json).

Xử lý: * Đọc hình ảnh và gửi prompt kèm câu hỏi tới API để nhận câu trả lời suy luận.

Tích hợp cơ chế Resume để tiếp tục lưu tiến trình nếu bị gián đoạn, tránh việc phải chạy lại từ đầu.

Đầu ra: Các file pixtral_ans_0.json và pixtral_ans_1.json chứa câu hỏi gốc và câu trả lời do AI sinh ra.

2. translate_q.ipynb - Dịch thuật và làm phong phú câu hỏi
Notebook này chuyển đổi các câu hỏi tiếng Anh sang tiếng Việt để chuẩn bị cho việc huấn luyện hoặc đánh giá các mô hình tiếng Việt.

Đầu vào: Kết quả từ bước 1 (pixtral_ans_0.json và pixtral_ans_1.json).

Xử lý:

Sử dụng thư viện deep_translator (Google Translator) để dịch tự động các câu hỏi.

Tích hợp từ điển paraphrase_map cho các câu hỏi phổ biến (ví dụ: "What is this?" có thể được dịch thành "Cái gì đây?", "Đây là vật gì?", "Món đồ này là gì?", v.v.). Việc này giúp làm phong phú và đa dạng hóa cấu trúc ngôn ngữ của tập dữ liệu.

Đầu ra: Các file đã được dịch tiếng Việt (pixtral_ans_0_translated.json và pixtral_ans_1_translated.json).

3. EDA.ipynb - Phân tích Khám phá Dữ liệu (EDA)
Notebook này tập trung vào việc đánh giá chất lượng và đặc điểm của tập dữ liệu sau khi đã được dịch và có câu trả lời. Phân tích được thực hiện để so sánh đặc trưng của tập câu hỏi không thể trả lời (unanswerable - 0) và tập có thể trả lời (answerable - 1).

Đầu vào: Dữ liệu hoàn chỉnh từ bước 2 (pixtral_ans_0_translated.json và pixtral_ans_1_translated.json).

Xử lý & Trực quan hóa:

Load dữ liệu JSON vào cấu trúc Pandas DataFrame.

Tính toán và so sánh chiều dài của câu hỏi và câu trả lời AI (số lượng từ).

Vẽ biểu đồ phân phối tần suất của các loại câu trả lời (answer_type) bằng seaborn.countplot.

Trực quan hóa phân phối độ dài văn bản qua các biểu đồ Histogram kết hợp đường KDE.

Thống kê Top 10 câu hỏi xuất hiện nhiều nhất trong tập dữ liệu để nắm bắt xu hướng từ vựng.

Yêu cầu thư viện (Dependencies)
Để chạy toàn bộ các notebook trong dự án, hệ thống cần được cài đặt các thư viện Python sau:

pandas, numpy

matplotlib, seaborn

tqdm

deep-translator

Bất kỳ client SDK nào được cấu hình cho việc gọi API (như Mistral/Pixtral).

Hướng dẫn sử dụng
Chạy gen_data.ipynb để xử lý tập ảnh và lấy câu trả lời từ AI.

Chạy translate_q.ipynb để dịch và đa dạng hóa các câu hỏi thu được ở bước 1 sang tiếng Việt.

Mở và chạy EDA.ipynb để xem các báo cáo thống kê và biểu đồ phân phối của dữ liệu cuối cùng, giúp đưa ra quyết định chọn model phù hợp cho bước tiếp theo.
