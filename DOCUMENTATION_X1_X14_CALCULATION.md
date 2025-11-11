# TÀI LIỆU: CÁCH TÍNH 14 CHỈ SỐ TÀI CHÍNH (X1-X14)

## 📊 **NGUỒN DỮ LIỆU**

Backend đọc dữ liệu từ file Excel với **3 sheets bắt buộc**:
1. **CDKT**: Cân đối kế toán (Balance Sheet)
2. **BCTN**: Báo cáo thu nhập (Income Statement)
3. **LCTT**: Lưu chuyển tiền tệ (Cash Flow Statement)

---

## 💰 **ĐƠN VỊ DỮ LIỆU**

⚠️ **QUAN TRỌNG**:
- Backend **KHÔNG** có code xử lý chuyển đổi đơn vị "tỷ VND"
- Tất cả số liệu trong Excel phải cùng đơn vị (VD: triệu VND, tỷ VND, đồng)
- Các chỉ số X1-X14 là **TỶ LỆ** (ratios) nên không bị ảnh hưởng bởi đơn vị **NẾU** các chỉ tiêu cùng đơn vị

**Ví dụ**:
```
X_1 = Lợi nhuận gộp / Doanh thu thuần

Trường hợp 1 (đơn vị: triệu VND):
- Lợi nhuận gộp = 500 triệu
- Doanh thu = 2,000 triệu
→ X_1 = 500/2000 = 0.25 ✅

Trường hợp 2 (đơn vị: tỷ VND):
- Lợi nhuận gộp = 0.5 tỷ
- Doanh thu = 2 tỷ
→ X_1 = 0.5/2 = 0.25 ✅

→ KẾT QUẢ GIỐNG NHAU!
```

---

## 📐 **CÔNG THỨC TÍNH 14 CHỈ SỐ**

### **X_1: Hệ số biên lợi nhuận gộp (Gross Profit Margin)**

```
X_1 = Lợi nhuận gộp (BCTN) / Doanh thu thuần (BCTN)
```

**Nguồn dữ liệu**:
- Lợi nhuận gộp: BCTN → "lợi nhuận gộp"
- Doanh thu thuần: BCTN → "doanh thu thuần" hoặc "doanh thu bán"

---

### **X_2: Hệ số biên lợi nhuận trước thuế (Profit Margin)**

```
X_2 = Lợi nhuận trước thuế (LCTT) / Doanh thu thuần (BCTN)
```

**Nguồn dữ liệu**:
- ⚠️ **Lợi nhuận trước thuế**: **LCTT** (không phải BCTN!)
- Doanh thu thuần: BCTN

---

### **X_3: Tỷ suất lợi nhuận trước thuế trên tổng tài sản (ROA)**

```
X_3 = Lợi nhuận trước thuế (LCTT) / Bình quân tổng tài sản (CDKT)
```

**Nguồn dữ liệu**:
- Lợi nhuận trước thuế: LCTT
- Bình quân tổng tài sản: CDKT → "tổng tài sản" → **Trung bình 2 cột cuối** (đầu kỳ + cuối kỳ) / 2

---

### **X_4: Tỷ suất lợi nhuận trước thuế trên vốn chủ sở hữu (ROE)**

```
X_4 = Lợi nhuận trước thuế (LCTT) / Bình quân vốn chủ sở hữu (CDKT)
```

**Nguồn dữ liệu**:
- Lợi nhuận trước thuế: LCTT
- Bình quân VCSH: CDKT → "vốn chủ sở hữu" → **Trung bình 2 cột cuối**

---

### **X_5: Hệ số nợ trên tài sản (Debt to Assets Ratio)**

```
X_5 = Nợ phải trả (CDKT) / Tổng tài sản (CDKT)
```

**Nguồn dữ liệu**:
- Nợ phải trả: CDKT → "nợ phải trả" hoặc "tổng nợ"
- Tổng tài sản: CDKT → "tổng tài sản" (cột cuối)

---

### **X_6: Hệ số nợ trên vốn chủ sở hữu (Debt to Equity Ratio)**

```
X_6 = Nợ phải trả (CDKT) / Vốn chủ sở hữu (CDKT)
```

**Nguồn dữ liệu**:
- Nợ phải trả: CDKT
- Vốn chủ sở hữu: CDKT (cột cuối)

---

### **X_7: Khả năng thanh toán hiện hành (Current Ratio)**

```
X_7 = Tài sản ngắn hạn (CDKT) / Nợ ngắn hạn (CDKT)
```

**Nguồn dữ liệu**:
- Tài sản ngắn hạn: CDKT → "tài sản ngắn hạn" (cột cuối)
- Nợ ngắn hạn: CDKT → "nợ ngắn hạn" (cột cuối)

---

### **X_8: Khả năng thanh toán nhanh (Quick Ratio)**

```
X_8 = (Tài sản ngắn hạn - Hàng tồn kho) / Nợ ngắn hạn
```

**Nguồn dữ liệu**:
- Tài sản ngắn hạn: CDKT (cột cuối)
- Hàng tồn kho: CDKT → "hàng tồn kho" (cột cuối)
- Nợ ngắn hạn: CDKT (cột cuối)

---

### **X_9: Hệ số khả năng trả lãi (Interest Coverage Ratio)**

```
X_9 = (Lợi nhuận trước thuế + Chi phí lãi vay) / Chi phí lãi vay
```

**Nguồn dữ liệu**:
- Lợi nhuận trước thuế: LCTT
- ⚠️ **Chi phí lãi vay**: **LCTT** (không phải BCTN!) → "chi phí lãi vay" hoặc "chi phí lãi" hoặc "lãi vay"

---

### **X_10: Hệ số khả năng trả nợ gốc (Debt Service Coverage Ratio)**

```
X_10 = (LNTT + Lãi vay + Khấu hao) / (Lãi vay + Nợ dài hạn)
```

**Nguồn dữ liệu**:
- LNTT: LCTT
- Lãi vay: LCTT
- ⚠️ **Khấu hao TSCĐ**: **LCTT** (không phải BCTN!) → "khấu hao tscđ" hoặc "khấu hao"
- Nợ dài hạn: CDKT → "nợ dài hạn" (cột cuối)

---

### **X_11: Hệ số khả năng tạo tiền trên vốn chủ sở hữu (Cash to Equity)**

```
X_11 = Tiền và tương đương tiền (CDKT) / Vốn chủ sở hữu (CDKT)
```

**Nguồn dữ liệu**:
- Tiền: CDKT → "tiền" hoặc "tiền và tương đương" (cột cuối)
- Vốn chủ sở hữu: CDKT (cột cuối)

---

### **X_12: Vòng quay hàng tồn kho (Inventory Turnover)**

```
X_12 = |Giá vốn hàng bán| / Bình quân hàng tồn kho
```

⚠️ **LƯU Ý**: Lấy **giá trị tuyệt đối** (chuyển âm thành dương)

**Nguồn dữ liệu**:
- Giá vốn hàng bán: BCTN → "giá vốn"
- Bình quân HTK: CDKT → "hàng tồn kho" → **Trung bình 2 cột cuối**

---

### **X_13: Kỳ thu tiền bình quân (Days Sales Outstanding - DSO)**

```
X_13 = 365 / (Doanh thu thuần / Bình quân phải thu)
```

Đơn giản hóa:
```
X_13 = 365 × Bình quân phải thu / Doanh thu thuần
```

**Nguồn dữ liệu**:
- Doanh thu thuần: BCTN
- Bình quân phải thu: CDKT → "phải thu" → **Trung bình 2 cột cuối**

---

### **X_14: Hiệu suất sử dụng tài sản (Asset Turnover)**

```
X_14 = Doanh thu thuần (BCTN) / Bình quân tổng tài sản (CDKT)
```

**Nguồn dữ liệu**:
- Doanh thu thuần: BCTN
- Bình quân tổng tài sản: CDKT → **Trung bình 2 cột cuối**

---

## 🔄 **QUY TRÌNH XỬ LÝ DỮ LIỆU**

### 1. **Đọc từ Excel** (`excel_processor.py:21-50`)

```python
def read_excel(file_path: str):
    # Đọc 3 sheets: CDKT, BCTN, LCTT
    # Kiểm tra sheets có đầy đủ không
```

### 2. **Lấy giá trị từ sheet** (`excel_processor.py:52-167`)

```python
def get_value_from_sheet(df, indicator_name, column_index=-1):
    # Tìm dòng chứa indicator_name (case-insensitive)
    # Lấy giá trị từ cột chỉ định:
    #   -1 = cột cuối (cuối kỳ)
    #   -2 = cột trước cuối (đầu kỳ)

    # XỬ LÝ FORMAT SỐ:
    # - "1,000,000.50" (US format)
    # - "1.000.000,50" (EU format)
    # - "(1000)" = số âm
    # - "-1000" = số âm

    # ⚠️ KHÔNG XỬ LÝ "tỷ VND" hay "triệu VND"!
```

### 3. **Tính bình quân 2 kỳ** (`excel_processor.py:169-191`)

```python
def get_average_from_two_periods(df, indicator_name):
    cuoi_ky = get_value_from_sheet(df, indicator_name, -1)
    dau_ky = get_value_from_sheet(df, indicator_name, -2)
    return (cuoi_ky + dau_ky) / 2
```

### 4. **Tính 14 chỉ số** (`excel_processor.py:193-316`)

```python
def calculate_14_indicators():
    # Lấy tất cả chỉ tiêu từ 3 sheets
    # Áp dụng công thức tính X_1 đến X_14
    # Làm tròn 6 chữ số thập phân
    # Return dict: {'X_1': 0.25, 'X_2': 0.08, ...}
```

---

## ⚠️ **VẤN ĐỀ VỚI ĐƠN VỊ "TỶ VND"**

### **Hiện trạng**:

❌ Backend **KHÔNG** có code xử lý chuyển đổi đơn vị:
- Không có check "tỷ" hay "triệu" trong cell
- Không có conversion factor (× 1000, ÷ 1000, etc.)
- Chỉ parse số từ string và return giá trị đó

### **Tại sao cần quan tâm?**

✅ **Nếu tất cả chỉ tiêu cùng đơn vị** → KHÔNG VẤN ĐỀ
- Ví dụ: Tất cả đều "tỷ VND" → X1-X14 đúng (vì là tỷ lệ)

❌ **Nếu các chỉ tiêu khác đơn vị** → SAI KẾT QUẢ
- Ví dụ:
  - Doanh thu: 100 tỷ
  - Lợi nhuận: 5,000 triệu (= 5 tỷ)
  - X_1 = 5000 / 100 = 50 (SAI! Phải là 0.05)

### **Trường hợp gây lỗi**:

1. **Cell có text "tỷ VND"**:
   ```
   Cell A2: "100 tỷ VND"
   ```
   → Code sẽ cố parse "tỷ" → **LỖI**!

2. **Cell có format đặc biệt**:
   ```
   Cell format: "Tỷ VND" (custom format)
   Value: 100
   Display: "100 Tỷ VND"
   ```
   → pandas đọc value = 100 → ✅ OK

3. **Đơn vị ghi trong header/row name**:
   ```
   Row: "Doanh thu thuần (tỷ VND)"
   Value: 100
   ```
   → Code vẫn tìm được dòng → ✅ OK

---

## 🔧 **CÁCH FIX NẾU CÓ VẤN ĐỀ**

### **Fix 1: Xử lý text "tỷ VND" trong cell**

Thêm vào `get_value_from_sheet()`:

```python
# Loại bỏ text "tỷ", "triệu", "VND" trước khi parse
value_str = value_str.replace('tỷ', '').replace('triệu', '').replace('VND', '')

# Nếu có "tỷ" → nhân lên 1000 để về triệu
if 'tỷ' in original_str.lower():
    float_value *= 1000  # chuyển tỷ → triệu
```

### **Fix 2: Chuyển đổi đơn vị tự động**

Thêm parameter `unit` vào `read_excel()`:

```python
def read_excel(file_path: str, input_unit: str = 'auto'):
    """
    input_unit: 'billion' (tỷ), 'million' (triệu), 'auto' (tự động phát hiện)
    """
    # Đọc Excel
    # Nếu input_unit = 'billion' → chia tất cả giá trị cho 1000
```

### **Fix 3: Thêm validation**

```python
def validate_data_consistency():
    """
    Kiểm tra consistency của dữ liệu:
    - Tổng tài sản = Nợ + VCSH
    - Lợi nhuận gộp = Doanh thu - Giá vốn
    Nếu sai → cảnh báo có thể do đơn vị không đồng nhất
    """
```

---

## 📝 **CÂU HỎI CHO BẠN**

Để tôi fix chính xác, vui lòng cho biết:

1. **Lỗi cụ thể là gì?**
   - Message lỗi đầy đủ?
   - Xuất hiện ở bước nào (upload file / tính toán)?

2. **Format dữ liệu trong Excel?**
   ```
   A. Cell có text "tỷ VND"?
      Ví dụ: "100 tỷ VND"

   B. Hay chỉ số không, format "Tỷ VND"?
      Ví dụ: 100 (display as "100 Tỷ VND")

   C. Hay đơn vị trong tên chỉ tiêu?
      Ví dụ: "Doanh thu (tỷ VND)" | 100
   ```

3. **"App" là gì?**
   - Frontend Vue.js (trong repo này)?
   - App mobile riêng?
   - Desktop app khác?

4. **Công thức trong "app" có khác backend không?**
   - App tính X1-X14 như thế nào?
   - Có file tài liệu công thức không?

---

## 📂 **FILES LIÊN QUAN**

- **`excel_processor.py`**: Xử lý Excel và tính X1-X14
  - `read_excel()`: Dòng 21-50
  - `get_value_from_sheet()`: Dòng 52-167
  - `calculate_14_indicators()`: Dòng 193-316

- **`main.py`**: API endpoints
  - `/process-excel`: Upload và tính chỉ số
  - `/train-survival`: Training model

---

Sau khi bạn cung cấp thông tin trên, tôi sẽ fix chính xác vấn đề!
