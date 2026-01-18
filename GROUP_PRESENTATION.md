# TÀI LIỆU THUYẾT TRÌNH NHÓM: DỰ ÁN AI PHÂN LOẠI LINH KIỆN ĐIỆN TỬ (MLP NUMPY)

Tài liệu này được biên soạn để đảm bảo nội dung thuyết trình của 3 thành viên liền mạch, logic và chuyên nghiệp.

---

## 📅 CẤU TRÚC BÀI THUYẾT TRÌNH (3 PHẦN - 1 CÂU CHUYỆN)

### 🎤 PHẦN 1: DỮ LIỆU & CƠ SỞ HẠ TẦNG (THÀNH VIÊN 1)
*Mục tiêu: Đặt vấn đề và giới thiệu "nguyên liệu" đầu vào.*

**1. Mở đầu & Đặt vấn đề:**
- Giới thiệu mục tiêu dự án: Tạo ra một hệ thống tự động nhận diện 10 loại linh kiện điện tử (Battery, Capacitor, Resistor, etc.) từ hình ảnh.
- Tại sao dùng NumPy? Để hiểu sâu bản chất toán học của Deep Learning mà không phụ thuộc vào các thư viện đen (black-box) như PyTorch/TensorFlow.

**2. Quy trình xử lý dữ liệu (The Kitchen):**
- **Thu thập:** Dữ liệu được tổ chức theo thư mục (Class-based structure).
- **Tiền xử lý (`load_dataset`):** 
    - Chuyển ảnh sang thang độ xám (`L`) để giảm khối lượng tính toán.
    - Resize về 64x64 pixel (tổng 4096 đầu vào).
    - Chuẩn hóa (Normalization) về khoảng [0, 1] bằng cách chia cho 255.0.

**3. Minh họa trực quan (Technical Preview - Đã tích hợp trong Demo):**
- **Ảnh tiền xử lý:** Show cho khán giả thấy sự khác biệt giữa ảnh gốc và ảnh 64x64 (mờ hơn, grayscale) - đây là những gì AI thực sự nhìn thấy.
- **Vectơ số (Vectorization):** Chuyển đổi từ ảnh sang một mảng NumPy chứa 4096 con số. Giải thích rằng máy tính không hiểu "hình ảnh", nó chỉ hiểu các giá trị cường độ sáng từ 0 đến 1.

**4. Công cụ trực quan & Ứng dụng (`streamlit_app.py`):**
- Giới thiệu giao diện web demo đã xây dựng: Upload ảnh -> Inference -> Hiện kết quả & Xác suất (Confidence).
- *Thực hiện Demo trực tiếp:* Tải một ảnh lên và chỉ vào phần **"Technical: Preprocessing & Vectorization"** để giải thích cho khán giả.
- *Lời dẫn chuyển giao:* "Sau khi đã có nguyên liệu sạch, chúng ta cần một bộ não để xử lý chúng. Sau đây, bạn [Tên Thành viên 2] sẽ giới thiệu về kiến trúc bộ não này."

---

### 🎤 PHẦN 2: KIẾN TRÚC MÔ HÌNH & LUỒNG TƯ DUY (THÀNH VIÊN 2)
*Mục tiêu: Giải thích cấu trúc "bộ não" và cách nó đưa ra dự đoán.*

**1. Cấu trúc mạng Nơ-ron (The Brain Architecture):**
- Giới thiệu mô hình MLP 6 tầng (5 tầng ẩn): 4096 (Input) -> 4096 -> 2048 -> 1024 -> 512 -> 256 -> 10 (Output).
- **Khởi tạo (`initialize_model`):** Sử dụng phương pháp **He Initialization** để các trọng số không quá lớn cũng không quá nhỏ, giúp mạng dễ học hơn.

**2. Luồng suy luận (`forward`):**
- Giải thích phép toán cốt lõi: `Y = X.W + b` (Dữ liệu nhân Trọng số cộng Độ lệch).
- **Hàm kích hoạt (Filter):**
    - `ReLU`: Đóng vai trò bộ lọc thông tin, loại bỏ các giá trị âm (không quan trọng) để giữ lại đặc trưng nổi bật.
    - `Softmax` (tại tầng cuối): Biến kết quả thô thành phần trăm xác suất (ví dụ: 90% là Resistor).

**3. Đo lường sai số (`cross_entropy`):**
- Cách mô hình tự đánh giá: So sánh dự đoán với nhãn thật. "Hình phạt" càng cao nếu mô hình càng tự tin vào đáp án sai.
- *Lời dẫn chuyển giao:* "Nhưng làm thế nào để mô hình biết mình sai ở đâu và tự sửa? Đây chính là phần tinh túy nhất do bạn [Tên Thành viên 3] trình bày."

---

### 🎤 PHẦN 3: CƠ CHẾ HỌC TẬP & TỐI ƯU HÓA (THÀNH VIÊN 3)
*Mục tiêu: Giải thích cách AI "rút kinh nghiệm" và kết quả đạt được.*

**1. Lan truyền ngược - Tìm lỗi (`backward`):**
- Đây là bước "Hồi tưởng": Đi ngược từ kết quả sai về từng lớp phía trước.
- Sử dụng đạo hàm (`relu_derivative`) để tính xem mỗi sợi dây thần kinh (W) đã đóng góp bao nhiêu phần vào cái sai đó.

**2. Cập nhật thông minh (`update_parameters`):**
- Thuật toán Gradient Descent: Điều chỉnh nhẹ các trọng số theo hướng giảm lỗi.
- **Learning Rate:** Giải thích tầm quan trọng của việc "học từ từ" để không bỏ lỡ điểm tối ưu.

**3. Huấn luyện & Kết quả (`train`, `test`):**
- Quy trình `Epoch`: Cho mô hình xem đi xem lại dữ liệu (30 vòng) để thẩm thấu kiến thức.
- **Shuffle:** Xáo trộn ảnh để mô hình không "học vẹt" thứ tự.
- **Trình diễn kết quả:** Show biểu đồ loss (giảm dần qua thời gian) và độ chính xác (Accuracy) cuối cùng trên tập Test.

**4. Kết luận:**
- Tóm tắt: Dự án đã xây dựng thành công bộ phân loại linh kiện từ con số 0 với NumPy.
- Hướng phát triển: Thử nghiệm với CNN (Mạng nơ-ron tích chập) hoặc tăng cường dữ liệu để chính xác hơn.

---

## 💡 MẸO ĐỂ LIỀN MẠCH TRONG BUỔI DIỄN
- **Sử dụng từ nối:** "Tiếp nối phần dữ liệu của...", "Như Thành viên 1 đã nói...", "Để cụ thể hóa kiến trúc mà Thành viên 2 vừa nêu...".
- **Ánh mắt:** Thành viên vừa kết thúc nên nhìn về phía thành viên sắp bắt đầu để dẫn dắt sự chú ý của khán giả.
- **Thống nhất thuật ngữ:** Cả nhóm dùng chung từ "Trọng số" (Weights), "Tầng ẩn" (Hidden layers), "Độ lỗi" (Loss).

---
*Tài liệu được soạn thảo tự động bởi Antigravity AI hỗ trợ nhóm của bạn.*
