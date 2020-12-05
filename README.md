# Project-1
File dữ liệu vượt quá dung lượng cho phép nên không thể upload lên được.

Các mô hình đều có các lớp encoder giúp trích xuất đặc trưng của ảnh và nén về độ phân giải nhỏ hơn để giảm tham số khi chạy qua lớp xử lí (LSTM, GRU, BiLSTM, ConvLSTM2D) sau đó decode ra để được ảnh cần dự đoán

# Thay đổi tới ngày 5-12

File ConvLSTM2D-3output.py là file code đùng model dự đoán 3 ảnh output kế tiếp hiện đang không thực hiện được việc chập 3 ảnh predict theo chiều ngang để so sánh với 3 ảnh output đã chập theo chiều ngang.

Thư mục 001 là một thư mục mẫu chứa 5 ảnh input (cat_[0-4].png) và 3 ảnh output (cat_[5-7].png) dùng để train/test model
