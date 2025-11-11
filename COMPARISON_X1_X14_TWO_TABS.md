# BÁO CÁO SO SÁNH CÁCH TÍNH X1-X14 GIỮA 2 TABS

## 🎯 **KẾT LUẬN CHÍNH**

✅ **CẢ 2 TABS SỬ DỤNG CÙNG CODE TÍNH TOÁN - KHÔNG CÓ SỰ KHÁC BIỆT**

**Tab "Dự báo PD"** và **Tab "Phân tích sống sót"** đều gọi cùng một function:

```python
excel_processor.calculate_14_indicators()
```

**Location**: `backend/excel_processor.py` dòng 193-316

---

## 📊 **CHỨNG MINH**

### **1. Tab "Dự báo PD"**

**Frontend**: `frontend/src/App.vue` dòng 3023
```javascript
const response = await axios.post(`${API_BASE}/predict-from-xlsx`, formData)
```

**Backend**: `backend/main.py` dòng 264-303
```python
@app.post("/predict-from-xlsx")
async def predict_from_xlsx(file: UploadFile = File(...)):
    # ...
    excel_processor.read_excel(tmp_file_path)       # Dòng 299
    indicators = excel_processor.calculate_14_indicators()  # Dòng 302 ✅
    # ...
```

---

### **2. Tab "Phân tích sống sót"**

**Frontend**: `frontend/src/App.vue` dòng 4744
```javascript
const response = await axios.post(`${API_BASE}/predict-survival`, formData)
```

**Backend**: `backend/main.py` dòng 2346-2427
```python
@app.post("/predict-survival")
async def predict_survival(file: Optional[UploadFile] = File(None), ...):
    # ...
    excel_processor.read_excel(tmp_file_path)       # Dòng 2376
    indicators = excel_processor.calculate_14_indicators()  # Dòng 2377 ✅
    # ...
```

---

## 📋 **CHI TIẾT CÁCH TÍNH TỪNG CHỈ SỐ**

### **Tên Biến và Sheet Lấy Dữ Liệu**

| Biến | Tên trong Code | Sheet | Dòng | Lưu ý |
|------|----------------|-------|------|-------|
| **Doanh thu thuần** | `doanh_thu_thuan` | **BCTN** | 204 | Fallback: "doanh thu bán" |
| **Lợi nhuận gộp** | `loi_nhuan_gop` | **BCTN** | 208 | |
| **Giá vốn hàng bán** | `gia_von_hang_ban` | **BCTN** | 209 | |
| **Lợi nhuận trước thuế** | `loi_nhuan_truoc_thue` | **LCTT** ⚠️ | 212 | ⚠️ LẤY TỪ LCTT, KHÔNG PHẢI BCTN! |
| **Tổng tài sản (cuối kỳ)** | `tong_tai_san` | **CDKT** | 216 | Cột cuối (-1) |
| **Bình quân tổng tài sản** | `binh_quan_tong_tai_san` | **CDKT** | 217 | Trung bình 2 cột cuối |
| **Vốn chủ sở hữu (cuối kỳ)** | `von_chu_so_huu` | **CDKT** | 219 | Cột cuối (-1) |
| **Bình quân VCSH** | `binh_quan_von_chu_so_huu` | **CDKT** | 220 | Trung bình 2 cột cuối |
| **Nợ phải trả** | `no_phai_tra` | **CDKT** | 222 | Fallback: "tổng nợ" |
| **Tài sản ngắn hạn** | `tai_san_ngan_han` | **CDKT** | 226 | Cột cuối (-1) |
| **Nợ ngắn hạn** | `no_ngan_han` | **CDKT** | 227 | Cột cuối (-1) |
| **Hàng tồn kho (cuối kỳ)** | `hang_ton_kho` | **CDKT** | 228 | Cột cuối (-1) |
| **Bình quân HTK** | `binh_quan_hang_ton_kho` | **CDKT** | 231 | Trung bình 2 cột cuối |
| **Chi phí lãi vay** 🔴 | `lai_vay` | **LCTT** ⚠️ | 234-238 | ⚠️ **QUAN TRỌNG - XEM BÊN DƯỚI** |
| **Nợ dài hạn** | `no_dai_han` | **CDKT** | 241 | Cột cuối (-1) |
| **Khấu hao TSCĐ** | `khau_hao` | **LCTT** ⚠️ | 244-248 | ⚠️ LẤY TỪ LCTT, KHÔNG PHẢI BCTN! |
| **Tiền và tương đương** | `tien_va_tuong_duong` | **CDKT** | 250-252 | Fallback: "tiền và tương đương" |
| **Phải thu (cuối kỳ)** | `khoan_phai_thu` | **CDKT** | 254 | Cột cuối (-1) |
| **Bình quân phải thu** | `binh_quan_phai_thu` | **CDKT** | 256 | Trung bình 2 cột cuối |

---

## 🔴 **QUAN TRỌNG: CHI PHÍ LÃI VAY**

**Code tìm kiếm** (`excel_processor.py` dòng 234-238):

```python
# ✅ THAY ĐỔI: Lấy "chi phí Lãi vay" từ LCTT thay vì BCTN
lai_vay = self.get_value_from_sheet(self.lctt_df, "chi phí lãi vay")
if lai_vay == 0:
    lai_vay = self.get_value_from_sheet(self.lctt_df, "chi phí lãi")
if lai_vay == 0:
    lai_vay = self.get_value_from_sheet(self.lctt_df, "lãi vay")
```

**Thứ tự ưu tiên tìm kiếm**:
1. ✅ **"chi phí lãi vay"** (LCTT)
2. ✅ **"chi phí lãi"** (LCTT) - nếu không tìm thấy (1)
3. ✅ **"lãi vay"** (LCTT) - nếu không tìm thấy (1) và (2)

**⚠️ LƯU Ý**:
- Code TÌM KIẾM theo thứ tự trên
- Tìm kiếm **case-insensitive** (không phân biệt hoa thường)
- Tìm kiếm **substring** (chứa chuỗi con)
- Ví dụ:
  - "Chi phí lãi vay ngắn hạn" → ✅ MATCH với "chi phí lãi vay"
  - "Chi phí lãi vay dài hạn" → ✅ MATCH với "chi phí lãi vay"
  - "Lãi vay phải trả" → ✅ MATCH với "lãi vay"

**🚨 VẤN ĐỀ TIỀM ẨN**:
Nếu trong sheet LCTT có NHIỀU dòng chứa "chi phí lãi vay" (ví dụ: "chi phí lãi vay ngắn hạn", "chi phí lãi vay dài hạn"), code sẽ lấy **DÒNG ĐẦU TIÊN** tìm được!

**Code tìm kiếm** (`excel_processor.py` dòng 88-93):
```python
mask = df[col_name].apply(normalize_text).str.contains(
    search_name, na=False, regex=False
)

if mask.any():
    value = df.loc[mask, value_col].iloc[0]  # ← Lấy DÒNG ĐẦU TIÊN
```

---

## 📐 **CÔNG THỨC 14 CHỈ SỐ**

### **X_1: Hệ số biên lợi nhuận gộp**
```
X_1 = Lợi nhuận gộp (BCTN) / Doanh thu thuần (BCTN)
```

### **X_2: Hệ số biên lợi nhuận trước thuế**
```
X_2 = Lợi nhuận trước thuế (LCTT) / Doanh thu thuần (BCTN)
```

### **X_3: ROA**
```
X_3 = Lợi nhuận trước thuế (LCTT) / Bình quân tổng tài sản (CDKT)
```

### **X_4: ROE**
```
X_4 = Lợi nhuận trước thuế (LCTT) / Bình quân VCSH (CDKT)
```

### **X_5: Hệ số nợ trên tài sản**
```
X_5 = Nợ phải trả (CDKT) / Tổng tài sản (CDKT)
```

### **X_6: Hệ số nợ trên VCSH**
```
X_6 = Nợ phải trả (CDKT) / Vốn chủ sở hữu (CDKT)
```

### **X_7: Khả năng thanh toán hiện hành**
```
X_7 = Tài sản ngắn hạn (CDKT) / Nợ ngắn hạn (CDKT)
```

### **X_8: Khả năng thanh toán nhanh**
```
X_8 = (Tài sản ngắn hạn - Hàng tồn kho) / Nợ ngắn hạn
     = (CDKT - CDKT) / CDKT
```

### **X_9: Hệ số khả năng trả lãi** 🔴
```
X_9 = (LNTT + Chi phí lãi vay) / Chi phí lãi vay
    = (LCTT + LCTT) / LCTT
```
⚠️ **Lưu ý**: Tất cả đều từ **LCTT**

### **X_10: Hệ số khả năng trả nợ gốc** 🔴
```
X_10 = (LNTT + Chi phí lãi vay + Khấu hao) / (Chi phí lãi vay + Nợ dài hạn)
     = (LCTT + LCTT + LCTT) / (LCTT + CDKT)
```
⚠️ **Lưu ý**: LNTT, Chi phí lãi vay, Khấu hao đều từ **LCTT**

### **X_11: Hệ số khả năng tạo tiền/VCSH**
```
X_11 = Tiền và tương đương (CDKT) / Vốn chủ sở hữu (CDKT)
```

### **X_12: Vòng quay hàng tồn kho**
```
X_12 = |Giá vốn hàng bán (BCTN) / Bình quân HTK (CDKT)|
```
⚠️ **Lưu ý**: Lấy **giá trị tuyệt đối** (abs)

### **X_13: Kỳ thu tiền bình quân**
```
X_13 = 365 / (Doanh thu thuần / Bình quân phải thu)
     = 365 × Bình quân phải thu / Doanh thu thuần
     = (CDKT × 365) / BCTN
```

### **X_14: Hiệu suất sử dụng tài sản**
```
X_14 = Doanh thu thuần (BCTN) / Bình quân tổng tài sản (CDKT)
```

---

## ⚠️ **CÁC ĐIỂM CẦN LƯU Ý**

### **1. Các chỉ tiêu LẤY TỪ LCTT (không phải BCTN)**

| Chỉ tiêu | Sheet | Comment trong Code |
|----------|-------|-------------------|
| Lợi nhuận trước thuế | **LCTT** | Dòng 211: "✅ THAY ĐỔI: Lấy từ LCTT thay vì BCTN" |
| Chi phí lãi vay | **LCTT** | Dòng 233: "✅ THAY ĐỔI: Lấy từ LCTT thay vì BCTN" |
| Khấu hao TSCĐ | **LCTT** | Dòng 243: "✅ THAY ĐỔI: Lấy từ LCTT thay vì BCTN" |

**Lý do**: Code đã được cập nhật để lấy từ LCTT (Lưu chuyển tiền tệ) thay vì BCTN

---

### **2. Tìm kiếm SUBSTRING (không phải exact match)**

Code tìm kiếm theo **substring** (chứa chuỗi con), KHÔNG phải exact match:

**Ví dụ**:
```python
# Tìm "lãi vay" sẽ match:
- "Lãi vay phải trả"          ✅
- "Chi phí lãi vay"            ✅
- "Chi phí lãi vay ngắn hạn"  ✅
- "Chi phí lãi vay dài hạn"   ✅
- "Tổng lãi vay trong kỳ"     ✅
```

**🚨 Nguy cơ lấy sai dòng**:
- Nếu có nhiều dòng match, code lấy **dòng đầu tiên**
- Ví dụ: Có cả "Chi phí lãi vay ngắn hạn" (dòng 10) và "Tổng chi phí lãi vay" (dòng 15)
  → Lấy dòng 10

---

### **3. Bình quân 2 kỳ**

Các chỉ tiêu sau sử dụng **bình quân 2 cột cuối** (đầu kỳ + cuối kỳ) / 2:

- Bình quân tổng tài sản (X_3, X_14)
- Bình quân VCSH (X_4)
- Bình quân HTK (X_12)
- Bình quân phải thu (X_13)

**Code**:
```python
def get_average_from_two_periods(df, indicator_name):
    cuoi_ky = get_value_from_sheet(df, indicator_name, column_index=-1)   # Cột cuối
    dau_ky = get_value_from_sheet(df, indicator_name, column_index=-2)    # Cột trước cuối
    return (cuoi_ky + dau_ky) / 2
```

---

### **4. Fallback Values**

Một số chỉ tiêu có **fallback** (giá trị dự phòng) nếu không tìm thấy:

| Chỉ tiêu | Tìm đầu tiên | Fallback |
|----------|-------------|----------|
| Doanh thu thuần | "doanh thu thuần" | "doanh thu bán" |
| Nợ phải trả | "nợ phải trả" | "tổng nợ" |
| Chi phí lãi vay | "chi phí lãi vay" | "chi phí lãi" → "lãi vay" |
| Khấu hao | "khấu hao tscđ" | "khấu hao" → "khấu hao tài sản" |
| Tiền | "tiền" | "tiền và tương đương" |

---

### **5. Xử lý giá trị âm**

**X_12**: Lấy **giá trị tuyệt đối** (abs)

```python
x12_value = gia_von_hang_ban / binh_quan_hang_ton_kho
indicators['X_12'] = abs(x12_value)  # Chuyển âm thành dương
```

---

## ✅ **KẾT LUẬN**

### **Trả lời câu hỏi của bạn**:

**Q: "Có lệch không giữa 2 tabs?"**

**A**: ❌ **KHÔNG CÓ LỆCH** - Cả 2 tabs sử dụng cùng code tính toán X1-X14.

---

### **Về "chi phí lãi vay" vs "chi phí lãi vay ngắn hạn"**:

**Q**: "Chi phí lãi vay và chi phí lãi vay ngắn hạn là khác nhau nhé"

**A**: ✅ **ĐÚNG** - Chúng khác nhau trong thực tế.

**Nhưng**:
- Code tìm kiếm theo **substring**
- Nếu tìm "chi phí lãi vay" → Sẽ match cả "chi phí lãi vay ngắn hạn"
- Code lấy **dòng đầu tiên** tìm được

**⚠️ Khuyến nghị**:
1. Trong file Excel LCTT, đặt tên chính xác:
   - "Chi phí lãi vay" (TỔNG) - để ở **TRÊN CÙNG**
   - "Chi phí lãi vay ngắn hạn" (CHI TIẾT) - để ở dưới

2. Hoặc chỉ có 1 dòng "Chi phí lãi vay" (tổng)

3. Nếu cần tách riêng, cần **sửa code** để tìm exact match:
```python
# Thay vì:
lai_vay = self.get_value_from_sheet(self.lctt_df, "chi phí lãi vay")

# Nên sửa thành:
lai_vay = self.get_value_from_sheet_exact(self.lctt_df, "Chi phí lãi vay")
```

---

## 📁 **TÀI LIỆU THAM KHẢO**

- **Dự báo PD - Frontend**: `frontend/src/App.vue` dòng 3010-3041
- **Dự báo PD - Backend**: `backend/main.py` dòng 264-320
- **Phân tích sống sót - Frontend**: `frontend/src/App.vue` dòng 4684-4760
- **Phân tích sống sót - Backend**: `backend/main.py` dòng 2346-2451
- **Code tính X1-X14**: `backend/excel_processor.py` dòng 193-316

---

**Ngày tạo**: 2025-11-11
**Người tạo**: Claude Code
**Phiên bản**: 1.0
