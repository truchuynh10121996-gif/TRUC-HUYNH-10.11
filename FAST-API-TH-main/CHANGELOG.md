# Changelog - Credit Risk Assessment System

## Version 2.0.0 (Latest) - 2025-11-05

### 🎨 Major UI/UX Updates

#### Giao diện mới với phong cách pastel hồng
- Màu nền chính: `rgba(255, 182, 193, 0.15)` kết hợp với màu trắng
- Thêm nhiều hiệu ứng lung linh: gradient animations, shimmer effects, floating animations
- Header mới đẹp mắt với tông màu hồng gradient động
- Hiệu ứng hover và transitions mượt mà trên tất cả components

#### Sidebar cho Huấn luyện mô hình
- Sidebar có thể ẩn/hiện bên trái màn hình
- Di chuyển toàn bộ tính năng huấn luyện mô hình vào sidebar
- Nút toggle với animation pulse-glow

### ⚡ Core Feature Changes

#### 1. Thay đổi cách nhập dữ liệu
**TRƯỚC:** Người dùng phải nhập thủ công 14 chỉ số tài chính (X1-X14)

**SAU:** Người dùng chỉ cần tải lên file XLSX chứa báo cáo tài chính:
- File XLSX phải có 3 sheets:
  - **CDKT**: Cân đối kế toán
  - **BCTN**: Báo cáo thu nhập
  - **LCTT**: Lưu chuyển tiền tệ
- Hệ thống tự động tính toán 14 chỉ số từ các báo cáo này

#### 2. Hiển thị 14 chỉ số tài chính
- Hiển thị đầy đủ 14 chỉ số đã tính toán với tên và giá trị
- Công thức tính:
  - **X_1**: Hệ số biên lợi nhuận gộp = Lợi nhuận gộp / Doanh thu thuần
  - **X_2**: Hệ số biên lợi nhuận trước thuế = Lợi nhuận trước thuế / Doanh thu thuần
  - **X_3**: ROA = Lợi nhuận trước thuế / Bình quân tổng tài sản
  - **X_4**: ROE = Lợi nhuận trước thuế / Bình quân vốn chủ sở hữu
  - **X_5**: Hệ số nợ trên tài sản = Nợ phải trả / Tổng tài sản
  - **X_6**: Hệ số nợ trên vốn CSH = Nợ phải trả / Vốn chủ sở hữu
  - **X_7**: Khả năng thanh toán hiện hành = Tài sản ngắn hạn / Nợ ngắn hạn
  - **X_8**: Khả năng thanh toán nhanh = (Tài sản ngắn hạn - Hàng tồn kho) / Nợ ngắn hạn
  - **X_9**: Hệ số khả năng trả lãi = (Lợi nhuận trước thuế + Lãi vay) / Lãi vay
  - **X_10**: Hệ số khả năng trả nợ gốc = (LNTT + Lãi vay + Khấu hao) / (Lãi vay + Nợ DH đến hạn)
  - **X_11**: Khả năng tạo tiền/VCSH = Tiền và tương đương / Vốn CSH
  - **X_12**: Vòng quay hàng tồn kho = Giá vốn hàng bán / Bình quân HTK
  - **X_13**: Kỳ thu tiền bình quân = 365 / (Doanh thu thuần / Khoản phải thu BQ)
  - **X_14**: Hiệu suất sử dụng tài sản = Doanh thu / Bình quân tổng tài sản

#### 3. Dashboard với 2 biểu đồ phân tích
- **Biểu đồ 1 (Bar Chart)**: Nhóm chỉ số Sinh lời & Đòn bẩy (X1-X6)
- **Biểu đồ 2 (Radar Chart)**: Nhóm chỉ số Thanh toán & Hiệu quả (X7-X14)

#### 4. Nâng cấp Gemini AI Analysis
- Phân tích dựa trên cả 14 chỉ số tài chính + PD từ 4 models
- Đưa ra khuyến nghị rõ ràng: **CHO VAY** hoặc **KHÔNG CHO VAY**
- Phân tích chi tiết từng nhóm chỉ số:
  - Khả năng sinh lời (X1-X4)
  - Đòn bẩy tài chính (X5-X6)
  - Khả năng thanh toán (X7-X8)
  - Khả năng trả nợ và tạo tiền (X9-X11)
  - Hiệu quả hoạt động (X12-X14)

#### 5. Xuất báo cáo Word
- Nút "📄 Xuất Báo cáo Word" xuất hiện sau khi có phân tích Gemini
- Báo cáo bao gồm:
  - Header với logo và tiêu đề
  - Kết quả dự báo PD từ 4 models
  - Bảng 14 chỉ số tài chính
  - 2 biểu đồ phân tích (PNG embedded)
  - Phân tích và khuyến nghị từ Gemini AI
  - Footer và disclaimer

### 🔧 Backend Updates

#### New Modules
- **excel_processor.py**: Module xử lý file XLSX và tính toán 14 chỉ số
- **report_generator.py**: Module tạo báo cáo Word với charts và analysis

#### New API Endpoints
- `POST /predict-from-xlsx`: Upload XLSX, tính 14 chỉ số và dự báo PD
- `POST /export-report`: Xuất báo cáo Word hoàn chỉnh

#### Updated Endpoints
- `POST /analyze`: Cập nhật để nhận và phân tích 14 chỉ số

#### New Dependencies
- `openpyxl==3.1.2`: Đọc file Excel
- `python-docx==1.1.0`: Tạo file Word
- `Pillow==10.2.0`: Xử lý ảnh
- `matplotlib==3.8.2`: Tạo biểu đồ

### 🗑️ Removed Features
- Removed: Form nhập thủ công 14 chỉ số
- Removed: Section cấu hình Gemini API Key trên giao diện
  - API Key giờ được cấu hình qua file `.env` trong backend

### 📋 Migration Guide

#### Từ Version 1.0.0 lên 2.0.0

1. **Cài đặt dependencies mới:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Cấu hình Gemini API Key:**
```bash
# Tạo file .env trong thư mục backend
cp .env.example .env
# Sửa GEMINI_API_KEY trong file .env
```

3. **Chuẩn bị dữ liệu:**
- Thay vì file CSV với 14 chỉ số, bạn cần file XLSX với 3 sheets (CDKT, BCTN, LCTT)
- Xem file mẫu trong thư mục `examples/` (nếu có)

4. **Khởi động lại hệ thống:**
```bash
# Backend
cd backend
python main.py

# Frontend
cd frontend
npm run dev
```

### 🐛 Bug Fixes
- Fixed: CORS issues khi gọi API từ frontend
- Fixed: Error handling khi file XLSX không đúng format
- Fixed: Memory leak khi generate báo cáo Word

### 🔮 Future Plans (v2.1.0)
- [ ] Support upload multiple files cùng lúc
- [ ] Add file validation trước khi upload
- [ ] Export báo cáo dạng PDF
- [ ] Dashboard analytics cho nhiều doanh nghiệp
- [ ] API authentication và user management
- [ ] Historical data tracking

---

## Version 1.0.0 - 2025-11-04

### Initial Release
- Basic credit risk assessment với Stacking Classifier
- Manual input của 14 chỉ số tài chính
- PD prediction từ 4 models: Logistic, Random Forest, XGBoost, Stacking
- Gemini AI analysis (basic)
- Vue.js frontend với Agribank theme (green)
- FastAPI backend
