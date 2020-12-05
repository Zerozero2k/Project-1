# Project-1
File dữ liệu vượt quá dung lượng cho phép nên không thể upload lên được.

Các mô hình đều có các lớp encoder giúp trích xuất đặc trưng của ảnh và nén về độ phân giải nhỏ hơn để giảm tham số khi chạy qua lớp xử lí (LSTM, GRU, BiLSTM, ConvLSTM2D) sau đó decode ra để được ảnh cần dự đoán

Mô hình dự đoán 3 ảnh đang không thực hiện được việc chập 3 ảnh predict theo chiều ngang để so sánh với 3 ảnh output đã chập theo chiều ngang.

File 001 là một thư mục mẫu trong các thư mục con dùng để train model
