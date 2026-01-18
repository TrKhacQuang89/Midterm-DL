# Kế Hoạch Phân Chia Công Việc Chi Tiết - Project MLP NumPy

Dựa trên cấu trúc file `mlp_network.py`, các phần kiến thức và mã nguồn được chia cụ thể cho 3 thành viên như sau:

---

## 👤 Thành Viên 1: Dữ Liệu & Hạ Tầng (Data & Infrastructure)
*Người này chịu trách nhiệm chuẩn bị "nguyên liệu" và hiển thị kết quả.*

**Các phần phụ trách trong code:**
- **Dataset Loading:** Hàm `load_dataset()`.
- **Model Storage:** Các hàm `save_model()` và `load_model()`.
- **Visualization:** Hàm `draw_loss()` và các thư viện `matplotlib`, `pandas`.

**Nội dung cần tìm hiểu:**
- Cách đọc ảnh từ thư mục, resize ảnh và chuẩn hóa pixel bằng thư viện **PIL**.
- Cách lưu trữ và tải trọng số mô hình sử dụng định dạng `.npz` của **NumPy**.
- Cách vẽ biểu đồ loss và accuracy để theo dõi quá trình huấn luyện.

---

## 👤 Thành Viên 2: Kiến Trúc & Lan Truyền Tiến (Architecture & Forward)
*Người này thiết kế "não bộ" của AI và quy định cách nó suy nghĩ.*

**Các phần phụ trách trong code:**
- **Model Initialization:** Hàm `initialize_model()`.
- **Activation Functions:** Các hàm `relu()` và `softmax()`.
- **Forward Pass:** Hàm `forward()`.
- **Loss Function:** Hàm `cross_entropy_loss()`.

**Nội dung cần tìm hiểu:**
- Cách khởi tạo ma trận trọng số (Weights) và độ lệch (Bias) (He Initialization).
- Cơ chế của hàm **ReLU** (lọc tín hiệu) và **Softmax** (tính xác suất lớp).
- Phép nhân ma trận giữa dữ liệu và trọng số (`np.dot`).
- Công thức tính độ lỗi Cross-Entropy giữa dự đoán và thực tế.

---

## 👤 Thành Viên 3: Toán Học & Tối Ưu Hóa (Math & Optimization)
*Người này chịu trách nhiệm cho cơ chế "học tập" của AI thông qua đạo hàm.*

**Các phần phụ trách trong code:**
- **Backward Pass (Quan trọng nhất):** Hàm `backward()`.
- **Derivatives:** Hàm `relu_derivative()`.
- **Parameter Update:** Hàm `update_parameters()`.
- **Execution Loop:** Các hàm `train()` và `test()`.

**Nội dung cần tìm hiểu:**
- Thuật toán **Backpropagation** (Lan truyền ngược) để tính lỗi cho từng lớp.
- Cách tính đạo hàm của hàm ReLU và Softmax.
- Thuật toán **Stochastic Gradient Descent (SGD)** để cập nhật trọng số.
- Cách điều chỉnh **Learning Rate** để mô hình hội tụ tốt nhất.

---

## 📈 Quy Trình Phối Hợp
1. **Thành viên 1** cung cấp danh sách ảnh (`data.append`) cho **Thành viên 3**.
2. **Thành viên 2** cung cấp cấu trúc mạng (`model`) cho **Thành viên 3**.
3. **Thành viên 3** điều khiển vòng lặp huấn luyện, sau đó chuyển kết quả cho **Thành viên 1** vẽ biểu đồ.

