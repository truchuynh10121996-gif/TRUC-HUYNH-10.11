"""
Gemini API Module - Tích hợp Google Gemini để phân tích kết quả dự báo PD
"""

import os
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Tải biến môi trường từ file .env

class GeminiAnalyzer:
    """Class để tích hợp Gemini API phân tích kết quả dự báo rủi ro tín dụng"""

    def __init__(self, api_key: str = None):
        """
        Khởi tạo Gemini API

        Args:
            api_key: API key của Google Gemini. Nếu không truyền, sẽ lấy từ biến môi trường GEMINI_API_KEY
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Không tìm thấy GEMINI_API_KEY. Vui lòng cung cấp API key hoặc set biến môi trường.")

        # Cấu hình Gemini
        genai.configure(api_key=self.api_key)

        # ✅ Sử dụng Gemini 2.0 Flash
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    def analyze_credit_risk(self, prediction_data: Dict[str, Any]) -> str:
        """
        Phân tích kết quả dự báo rủi ro tín dụng bằng Gemini

        Args:
            prediction_data: Dict chứa thông tin dự báo (PD, chỉ số tài chính, v.v.)

        Returns:
            Kết quả phân tích dạng text từ Gemini
        """
        # Tạo prompt chi tiết
        prompt = self._create_analysis_prompt(prediction_data)

        try:
            # Gọi Gemini API với self.model
            response = self.model.generate_content(prompt)
            result = response.text
            return result

        except Exception as e:
            return f"❌ Lỗi khi gọi Gemini API: {str(e)}"

    def _create_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """
        Tạo prompt chi tiết để gửi tới Gemini

        Args:
            data: Dữ liệu dự báo bao gồm PD và 14 chỉ số tài chính

        Returns:
            Prompt string
        """
        # Lấy thông tin PD
        prediction = data.get('prediction', {})
        pd_stacking = prediction.get('pd_stacking', 0) * 100
        pd_logistic = prediction.get('pd_logistic', 0) * 100
        pd_rf = prediction.get('pd_random_forest', 0) * 100
        pd_xgboost = prediction.get('pd_xgboost', 0) * 100
        prediction_label = prediction.get('prediction_label', 'N/A')

        # Lấy 14 chỉ số
        indicators_dict = data.get('indicators_dict', {})

        # Phân loại rủi ro theo 5 cấp độ
        if pd_stacking < 2:
            risk_level = "RỦI RO RẤT THẤP 🟢 (AAA-AA)"
            risk_desc = "doanh nghiệp xuất sắc, tình hình tài chính rất tốt"
            rating = "AAA-AA"
        elif pd_stacking < 5:
            risk_level = "RỦI RO THẤP 🟢 (A-BBB)"
            risk_desc = "doanh nghiệp tốt, tình hình tài chính ổn định"
            rating = "A-BBB"
        elif pd_stacking < 10:
            risk_level = "RỦI RO TRUNG BÌNH 🟡 (BB)"
            risk_desc = "doanh nghiệp cần theo dõi thêm"
            rating = "BB"
        elif pd_stacking < 20:
            risk_level = "RỦI RO CAO 🟠 (B)"
            risk_desc = "doanh nghiệp có rủi ro đáng kể, cần thận trọng"
            rating = "B"
        else:
            risk_level = "RỦI RO RẤT CAO 🔴 (CCC-D)"
            risk_desc = "doanh nghiệp có nguy cơ vỡ nợ rất cao"
            rating = "CCC-D"

        # Tạo chuỗi hiển thị 14 chỉ số
        indicators_str = f"""
X_1 (Hệ số biên lợi nhuận gộp): {indicators_dict.get('X_1', 0):.4f}
X_2 (Hệ số biên lợi nhuận trước thuế): {indicators_dict.get('X_2', 0):.4f}
X_3 (ROA): {indicators_dict.get('X_3', 0):.4f}
X_4 (ROE): {indicators_dict.get('X_4', 0):.4f}
X_5 (Hệ số nợ trên tài sản): {indicators_dict.get('X_5', 0):.4f}
X_6 (Hệ số nợ trên vốn CSH): {indicators_dict.get('X_6', 0):.4f}
X_7 (Khả năng thanh toán hiện hành): {indicators_dict.get('X_7', 0):.4f}
X_8 (Khả năng thanh toán nhanh): {indicators_dict.get('X_8', 0):.4f}
X_9 (Hệ số khả năng trả lãi): {indicators_dict.get('X_9', 0):.4f}
X_10 (Hệ số khả năng trả nợ gốc): {indicators_dict.get('X_10', 0):.4f}
X_11 (Khả năng tạo tiền/Vốn CSH): {indicators_dict.get('X_11', 0):.4f}
X_12 (Vòng quay hàng tồn kho): {indicators_dict.get('X_12', 0):.4f}
X_13 (Kỳ thu tiền bình quân - ngày): {indicators_dict.get('X_13', 0):.2f}
X_14 (Hiệu suất sử dụng tài sản): {indicators_dict.get('X_14', 0):.4f}
"""

        prompt = f"""
Bạn là một chuyên gia phân tích rủi ro tín dụng của Agribank với hơn 20 năm kinh nghiệm.

Dựa trên kết quả dự báo xác suất vỡ nợ (PD) từ mô hình AI Stacking Classifier và 14 chỉ số tài chính của doanh nghiệp, hãy phân tích chi tiết và đưa ra khuyến nghị rõ ràng.

**HỆ THỐNG PHÂN LOẠI TÍN DỤNG (5 CẤP ĐỘ):**
- < 2%: Rất thấp (AAA-AA) - Doanh nghiệp xuất sắc
- 2-5%: Thấp (A-BBB) - Doanh nghiệp tốt
- 5-10%: Trung bình (BB) - Cần theo dõi
- 10-20%: Cao (B) - Rủi ro đáng kể
- > 20%: Rất cao (CCC-D) - Nguy cơ vỡ nợ cao

**KẾT QUẢ DỰ BÁO:**
- Xác suất Vỡ nợ (PD) - Stacking Model: {pd_stacking:.2f}%
- Xác suất Vỡ nợ (PD) - Logistic Regression: {pd_logistic:.2f}%
- Xác suất Vỡ nợ (PD) - Random Forest: {pd_rf:.2f}%
- Xác suất Vỡ nợ (PD) - XGBoost: {pd_xgboost:.2f}%
- Dự đoán: {prediction_label}
- Mức độ rủi ro: {risk_level}
- Credit Rating: {rating}

**14 CHỈ SỐ TÀI CHÍNH:**
{indicators_str}

**YÊU CẦU PHÂN TÍCH:**

Hãy phân tích theo cấu trúc sau (bằng tiếng Việt, chuyên nghiệp):

1. **Tổng quan rủi ro**: Đánh giá tổng thể về tình hình tài chính và khả năng trả nợ của doanh nghiệp

2. **Phân tích 14 chỉ số**:
   - Đánh giá các chỉ số khả năng sinh lời (X_1, X_2, X_3, X_4)
   - Phân tích khả năng thanh toán và đòn bẩy tài chính (X_5, X_6, X_7, X_8)
   - Đánh giá khả năng trả nợ và tạo tiền (X_9, X_10, X_11)
   - Phân tích hiệu quả hoạt động (X_12, X_13, X_14)
   - Chỉ ra những chỉ số TỐT và chỉ số CẦN CẢI THIỆN

3. **So sánh PD từ 4 models**:
   - Mức độ đồng thuận giữa các models
   - Giải thích sự khác biệt (nếu có)

4. **KHUYẾN NGHỊ CUỐI CÙNG** (QUAN TRỌNG):
   - Quyết định: **CHO VAY** hoặc **KHÔNG CHO VAY**
   - Giải thích lý do quyết định
   - Nếu cho vay: Đề xuất điều kiện và hạn mức phù hợp
   - Nếu không cho vay: Đề xuất doanh nghiệp cần cải thiện gì

5. **Lưu ý**: Các yếu tố cần theo dõi và giám sát

Hãy trình bày rõ ràng, dễ hiểu, có cấu trúc. Tối đa 500 từ.
"""

        return prompt

    def fetch_industry_data(self, industry: str, industry_name: str) -> Dict[str, Any]:
        """
        Lấy dữ liệu ngành nghề mới nhất từ AI

        Args:
            industry: Mã ngành
            industry_name: Tên ngành đầy đủ

        Returns:
            Dict chứa dữ liệu ngành nghề
        """
        prompt = f"""
Bạn là một chuyên gia kinh tế có quyền truy cập vào các nguồn dữ liệu thời gian thực.

Hãy thu thập và tổng hợp dữ liệu mới nhất (2024-2025) về ngành "{industry_name}" tại Việt Nam.

**YÊU CẦU DỮ LIỆU:**

1. **Chỉ số tăng trưởng:**
   - Tăng trưởng GDP ngành (5 năm gần nhất: 2020, 2021, 2022, 2023, 2024)
   - Tốc độ tăng trưởng doanh thu trung bình
   - Quy mô thị trường (tỷ USD)

2. **Chỉ số tài chính:**
   - ROE trung bình ngành
   - ROA trung bình ngành
   - Biên lợi nhuận gộp trung bình
   - Tỷ lệ nợ trên tổng tài sản trung bình

3. **Chỉ số rủi ro tín dụng:**
   - Tỷ lệ nợ xấu (NPL) của ngành (5 năm gần nhất)
   - Tỷ lệ vỡ nợ trung bình
   - Xếp hạng rủi ro ngành

4. **Các chỉ số khác:**
   - Số lượng doanh nghiệp trong ngành
   - Mức độ tập trung thị trường (HHI nếu có)
   - Xu hướng giá cả/chi phí

Trả về dữ liệu dưới dạng JSON với cấu trúc rõ ràng. Sử dụng số liệu thực nếu có, hoặc ước tính hợp lý dựa trên xu hướng.

Ví dụ format JSON:
{{
  "growth": {{
    "gdp_growth": [3.5, 4.2, 5.1, 6.0, 5.8],
    "years": [2020, 2021, 2022, 2023, 2024],
    "revenue_growth": 5.5,
    "market_size_usd": 50.2
  }},
  "financial": {{
    "roe": 12.5,
    "roa": 8.2,
    "gross_margin": 25.3,
    "debt_ratio": 45.6
  }},
  "credit_risk": {{
    "npl_rates": [2.1, 2.0, 1.8, 1.5, 1.4],
    "default_rate": 1.2,
    "risk_rating": "Trung bình"
  }},
  "other": {{
    "num_companies": 15000,
    "market_concentration": "Phân tán",
    "price_trend": "Tăng nhẹ"
  }}
}}
"""
        try:
            response = self.model.generate_content(prompt)
            data_text = response.text

            # Parse JSON từ response
            import json
            import re

            # Tìm JSON block trong response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', data_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                # Nếu không tìm thấy JSON, tạo dữ liệu mẫu
                data = self._generate_sample_data(industry_name)

            return {
                "status": "success",
                "data": data,
                "raw_text": data_text
            }

        except Exception as e:
            # Trường hợp lỗi, trả về dữ liệu mẫu
            return {
                "status": "fallback",
                "data": self._generate_sample_data(industry_name),
                "error": str(e)
            }

    def _generate_sample_data(self, industry_name: str) -> Dict[str, Any]:
        """Tạo dữ liệu mẫu cho testing"""
        return {
            "growth": {
                "gdp_growth": [3.5, 4.2, 5.1, 6.0, 5.8],
                "years": [2020, 2021, 2022, 2023, 2024],
                "revenue_growth": 5.5,
                "market_size_usd": 50.2
            },
            "financial": {
                "roe": 12.5,
                "roa": 8.2,
                "gross_margin": 25.3,
                "debt_ratio": 45.6
            },
            "credit_risk": {
                "npl_rates": [2.1, 2.0, 1.8, 1.5, 1.4],
                "default_rate": 1.2,
                "risk_rating": "Trung bình"
            },
            "other": {
                "num_companies": 15000,
                "market_concentration": "Phân tán",
                "price_trend": "Tăng nhẹ"
            }
        }

    def generate_charts_data(self, industry: str, industry_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tạo config biểu đồ ECharts từ dữ liệu và phân tích sơ bộ

        Args:
            industry: Mã ngành
            industry_name: Tên ngành
            data: Dữ liệu ngành từ fetch_industry_data

        Returns:
            Dict chứa charts_data (ECharts config) và brief_analysis
        """
        # Tạo nhiều loại biểu đồ ECharts
        charts_data = []

        # 1. Biểu đồ cột - Tăng trưởng GDP
        growth = data.get("growth", {})
        charts_data.append({
            "title": {"text": f"Tăng trưởng GDP - {industry_name}", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": growth.get("years", [])},
            "yAxis": {"type": "value", "name": "Tăng trưởng (%)"},
            "series": [{
                "data": growth.get("gdp_growth", []),
                "type": "bar",
                "itemStyle": {"color": "#FF6B9D"},
                "label": {"show": True, "position": "top"}
            }]
        })

        # 2. Biểu đồ Radar - Chỉ số tài chính
        financial = data.get("financial", {})
        charts_data.append({
            "title": {"text": f"Chỉ số Tài chính - {industry_name}", "left": "center"},
            "tooltip": {},
            "radar": {
                "indicator": [
                    {"name": "ROE", "max": 30},
                    {"name": "ROA", "max": 20},
                    {"name": "Biên LN gộp", "max": 50},
                    {"name": "Tỷ lệ nợ", "max": 100}
                ]
            },
            "series": [{
                "type": "radar",
                "data": [{
                    "value": [
                        financial.get("roe", 0),
                        financial.get("roa", 0),
                        financial.get("gross_margin", 0),
                        financial.get("debt_ratio", 0)
                    ],
                    "name": "Chỉ số tài chính",
                    "areaStyle": {"color": "rgba(255, 107, 157, 0.3)"}
                }]
            }]
        })

        # 3. Biểu đồ đường - Tỷ lệ nợ xấu
        credit_risk = data.get("credit_risk", {})
        charts_data.append({
            "title": {"text": f"Tỷ lệ Nợ xấu (NPL) - {industry_name}", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": growth.get("years", [])},
            "yAxis": {"type": "value", "name": "NPL (%)"},
            "series": [{
                "data": credit_risk.get("npl_rates", []),
                "type": "line",
                "smooth": True,
                "itemStyle": {"color": "#9C27B0"},
                "areaStyle": {"color": "rgba(156, 39, 176, 0.2)"},
                "label": {"show": True, "position": "top"}
            }]
        })

        # Phân tích sơ bộ bằng AI
        prompt = f"""
Dựa trên dữ liệu sau về ngành "{industry_name}":

{str(data)}

Hãy phân tích sơ bộ (200-300 từ) bằng tiếng Việt:

1. **Điểm nổi bật**: Những chỉ số tích cực/tiêu cực nhất
2. **Xu hướng**: Ngành đang phát triển hay suy giảm?
3. **Rủi ro tín dụng**: Đánh giá sơ bộ về NPL và khả năng trả nợ
4. **Nhận xét chung**: Đánh giá tổng thể về tình hình ngành

Trình bày ngắn gọn, súc tích, dễ hiểu.
"""

        try:
            response = self.model.generate_content(prompt)
            brief_analysis = response.text
        except Exception as e:
            brief_analysis = f"Không thể tạo phân tích sơ bộ. Lỗi: {str(e)}"

        return {
            "status": "success",
            "charts_data": charts_data,
            "brief_analysis": brief_analysis
        }

    def deep_analyze_industry(self, industry: str, industry_name: str, data: Dict[str, Any], brief_analysis: str) -> str:
        """
        Phân tích sâu ảnh hưởng của ngành đến quyết định cho vay

        Args:
            industry: Mã ngành
            industry_name: Tên ngành
            data: Dữ liệu ngành
            brief_analysis: Phân tích sơ bộ

        Returns:
            Phân tích sâu về ảnh hưởng đến quyết định tín dụng
        """
        prompt = f"""
Bạn là chuyên gia tín dụng cấp cao của Agribank với 20 năm kinh nghiệm.

Dựa trên dữ liệu và phân tích sơ bộ về ngành "{industry_name}", hãy đưa ra phân tích sâu về ảnh hưởng đến quyết định cho vay.

**DỮ LIỆU NGÀNH:**
{str(data)}

**PHÂN TÍCH SƠ BỘ:**
{brief_analysis}

**YÊU CẦU PHÂN TÍCH SÂU (400-500 từ):**

1. **Đánh giá rủi ro tín dụng ngành** (150 từ):
   - Phân tích chỉ số NPL và xu hướng
   - So sánh với trung bình toàn hệ thống ngân hàng (NPL VN ~2%)
   - Đánh giá mức độ rủi ro: Thấp/Trung bình/Cao

2. **Ảnh hưởng đến quyết định cho vay** (150 từ):
   - Ngành có phù hợp để cho vay không? Tại sao?
   - Điều kiện kinh tế vĩ mô ảnh hưởng như thế nào?
   - Chính sách Nhà nước/NHNN có thuận lợi không?

3. **Khuyến nghị cụ thể cho Agribank** (150 từ):
   - Có nên cho vay doanh nghiệp trong ngành này không?
   - Mức độ thận trọng: Bình thường/Thận trọng/Rất thận trọng
   - Điều kiện cho vay đề xuất:
     * Hạn mức: Thấp/Trung bình/Cao
     * Lãi suất: Ưu đãi/Tiêu chuẩn/Cao hơn
     * Thời hạn vay: Ngắn hạn/Trung hạn/Dài hạn
     * Tài sản đảm bảo: Yêu cầu hay không?
   - Các tiêu chí đánh giá riêng cho ngành này

4. **Lưu ý đặc biệt**: Các rủi ro tiềm ẩn cần theo dõi

**QUAN TRỌNG**: Phân tích phải thực tế, dựa trên dữ liệu, và đưa ra khuyến nghị CỤ THỂ, RÕ RÀNG.

Trả lời bằng tiếng Việt, chuyên nghiệp.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Lỗi khi phân tích sâu: {str(e)}"

    def analyze_pd_with_industry(self, indicators_dict: Dict[str, float], industry: str, industry_name: str) -> str:
        """
        Phân tích PD kết hợp với ngành nghề - tạo biểu đồ và phân tích chuyên sâu

        Args:
            indicators_dict: Dict chứa 14 chỉ số tài chính
            industry: Mã ngành
            industry_name: Tên ngành

        Returns:
            Phân tích chuyên sâu từ Gemini
        """
        # Tạo chuỗi hiển thị 14 chỉ số
        indicators_str = f"""
X_1 (Hệ số biên lợi nhuận gộp): {indicators_dict.get('X_1', 0):.4f}
X_2 (Hệ số biên lợi nhuận trước thuế): {indicators_dict.get('X_2', 0):.4f}
X_3 (ROA): {indicators_dict.get('X_3', 0):.4f}
X_4 (ROE): {indicators_dict.get('X_4', 0):.4f}
X_5 (Hệ số nợ trên tài sản): {indicators_dict.get('X_5', 0):.4f}
X_6 (Hệ số nợ trên vốn CSH): {indicators_dict.get('X_6', 0):.4f}
X_7 (Khả năng thanh toán hiện hành): {indicators_dict.get('X_7', 0):.4f}
X_8 (Khả năng thanh toán nhanh): {indicators_dict.get('X_8', 0):.4f}
X_9 (Hệ số khả năng trả lãi): {indicators_dict.get('X_9', 0):.4f}
X_10 (Hệ số khả năng trả nợ gốc): {indicators_dict.get('X_10', 0):.4f}
X_11 (Khả năng tạo tiền/Vốn CSH): {indicators_dict.get('X_11', 0):.4f}
X_12 (Vòng quay hàng tồn kho): {indicators_dict.get('X_12', 0):.4f}
X_13 (Kỳ thu tiền bình quân - ngày): {indicators_dict.get('X_13', 0):.2f}
X_14 (Hiệu suất sử dụng tài sản): {indicators_dict.get('X_14', 0):.4f}
"""

        prompt = f"""
Bạn là chuyên gia phân tích tín dụng của Agribank với 20 năm kinh nghiệm.

Dựa trên 14 chỉ số tài chính của doanh nghiệp và ngành nghề "{industry_name}", hãy phân tích chuyên sâu ảnh hưởng đến quyết định cho vay.

**14 CHỈ SỐ TÀI CHÍNH CỦA DOANH NGHIỆP:**
{indicators_str}

**NGÀNH NGHỀ:** {industry_name}

**YÊU CẦU PHÂN TÍCH (500-600 từ):**

1. **So sánh chỉ số doanh nghiệp với trung bình ngành** (150 từ):
   - Đánh giá các chỉ số sinh lời (X1-X4) so với ngành
   - Đánh giá đòn bẩy tài chính (X5-X6) so với ngành
   - Đánh giá khả năng thanh toán (X7-X8) so với ngành
   - Đánh giá hiệu quả hoạt động (X9-X14) so với ngành

2. **Phân tích rủi ro ngành kết hợp với tình hình doanh nghiệp** (200 từ):
   - Doanh nghiệp có phù hợp với đặc thù ngành không?
   - Những rủi ro đặc thù của ngành ảnh hưởng như thế nào?
   - Doanh nghiệp có khả năng chống chịu với rủi ro ngành không?
   - Xu hướng ngành có thuận lợi cho doanh nghiệp không?

3. **Khuyến nghị cho vay cụ thể** (150 từ):
   - **QUYẾT ĐỊNH**: CHO VAY / KHÔNG CHO VAY / CHO VAY CÓ ĐIỀU KIỆN
   - **Hạn mức đề xuất**: Cụ thể (VD: 5-10 tỷ, 10-20 tỷ, > 20 tỷ)
   - **Lãi suất**: Ưu đãi / Tiêu chuẩn / Cao hơn (bao nhiêu %)
   - **Thời hạn vay**: Ngắn hạn (< 1 năm) / Trung hạn (1-5 năm) / Dài hạn (> 5 năm)
   - **Tài sản đảm bảo**: Yêu cầu / Không yêu cầu, tỷ lệ TSBĐ/Hạn mức
   - **Điều kiện đặc biệt** (nếu có)

4. **Các chỉ số cần theo dõi đặc biệt** (100 từ):
   - Chỉ số nào cần theo dõi sát sao?
   - Tần suất kiểm tra đề xuất
   - Ngưỡng cảnh báo

**QUAN TRỌNG**:
- Phân tích phải CỤ THỂ, SỐ LIỆU, RÕ RÀNG
- Khuyến nghị phải có giá trị thực tiễn cho Agribank
- Tập trung vào ẢNH HƯỞNG CỦA NGÀNH đến quyết định

Trả lời bằng tiếng Việt, chuyên nghiệp.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Lỗi khi phân tích PD kết hợp: {str(e)}"

    def analyze_industry(self, industry: str, industry_name: str) -> Dict[str, Any]:
        """
        Phân tích tình hình ngành nghề và tác động đến quyết định cho vay

        Args:
            industry: Mã ngành (e.g., 'agriculture', 'finance')
            industry_name: Tên ngành đầy đủ

        Returns:
            Dict chứa phân tích và dữ liệu charts (nếu có)
        """
        prompt = f"""
Bạn là một chuyên gia kinh tế và phân tích ngành nghề của Agribank với hơn 20 năm kinh nghiệm.

Hãy phân tích chi tiết về ngành "{industry_name}" tại Việt Nam và tác động của nó đến quyết định cho vay của ngân hàng.

**YÊU CẦU PHÂN TÍCH:**

1. **Tổng quan ngành** (150 từ):
   - Tình hình hiện tại của ngành
   - Xu hướng phát triển gần đây (2023-2024)
   - Quy mô thị trường và tốc độ tăng trưởng
   - Các doanh nghiệp hàng đầu trong ngành

2. **Phân tích kinh tế vĩ mô** (150 từ):
   - Các chỉ số kinh tế quan trọng ảnh hưởng đến ngành
   - Chính sách của Chính phủ/Ngân hàng Nhà nước liên quan
   - Tác động của kinh tế toàn cầu
   - Lạm phát, lãi suất, tỷ giá ảnh hưởng như thế nào

3. **Cơ hội và Rủi ro** (150 từ):
   - Cơ hội: Những yếu tố tích cực cho ngành
   - Rủi ro: Những thách thức và nguy cơ tiềm ẩn
   - Đánh giá mức độ rủi ro của ngành (Thấp/Trung bình/Cao)

4. **Tác động đến quyết định cho vay** (150 từ):
   - Ngành này có phù hợp để cho vay không?
   - Các tiêu chí đánh giá khi cho vay doanh nghiệp trong ngành này
   - Mức độ rủi ro tín dụng của ngành (dựa trên NPL, tỷ lệ nợ xấu)
   - Khuyến nghị về hạn mức, lãi suất, và thời hạn vay phù hợp

5. **Dự báo và Khuyến nghị** (100 từ):
   - Triển vọng ngành trong 1-2 năm tới
   - Khuyến nghị chiến lược cho ngân hàng khi cho vay ngành này
   - Các điểm cần đặc biệt lưu ý

**LƯU Ý:**
- Sử dụng số liệu cụ thể, tham khảo các nguồn uy tín (GSO, World Bank, IMF, các báo cáo ngành...)
- Phân tích phải khách quan, dựa trên dữ liệu thực tế
- Trình bày rõ ràng, có cấu trúc, dễ hiểu
- Tổng số từ: khoảng 700 từ

Hãy trả lời bằng tiếng Việt, chuyên nghiệp và chi tiết.
"""

        try:
            response = self.model.generate_content(prompt)
            analysis = response.text

            # Tạo dữ liệu charts giả (trong thực tế có thể lấy từ API thực)
            charts = [
                {
                    "title": f"Tăng trưởng GDP ngành {industry_name} (2020-2024)",
                    "description": "Biểu đồ thể hiện tốc độ tăng trưởng GDP của ngành qua các năm"
                },
                {
                    "title": f"Tỷ lệ nợ xấu ngành {industry_name}",
                    "description": "Biểu đồ so sánh tỷ lệ NPL của ngành với trung bình toàn hệ thống"
                },
                {
                    "title": f"Doanh thu và Lợi nhuận ngành {industry_name}",
                    "description": "Xu hướng doanh thu và lợi nhuận của các doanh nghiệp trong ngành"
                }
            ]

            return {
                "analysis": analysis,
                "charts": charts
            }

        except Exception as e:
            return {
                "analysis": f"❌ Lỗi khi phân tích ngành: {str(e)}",
                "charts": []
            }

    def analyze_scenario_simulation(self, data: Dict[str, Any]) -> str:
        """
        Phân tích chuyên sâu kết quả mô phỏng kịch bản xấu

        Args:
            data: Dict chứa:
                - scenario_info: Thông tin kịch bản
                - indicators_before_dict: 14 chỉ số trước khi áp kịch bản
                - indicators_after_dict: 14 chỉ số sau khi áp kịch bản
                - prediction_before: PD trước khi áp kịch bản
                - prediction_after: PD sau khi áp kịch bản
                - pd_change: Thông tin thay đổi PD

        Returns:
            Kết quả phân tích dạng text từ Gemini
        """
        scenario_info = data.get('scenario_info', {})
        indicators_before = data.get('indicators_before_dict', {})
        indicators_after = data.get('indicators_after_dict', {})
        prediction_before = data.get('prediction_before', {})
        prediction_after = data.get('prediction_after', {})
        pd_change = data.get('pd_change', {})

        # Lấy thông tin PD
        pd_before = pd_change.get('before', 0) * 100
        pd_after = pd_change.get('after', 0) * 100
        pd_change_pct = pd_change.get('change_pct', 0)

        # Phân loại mức độ ảnh hưởng
        if abs(pd_change_pct) < 10:
            impact_level = "ẢNH HƯỞNG NHẸ 🟢"
            impact_desc = "Doanh nghiệp có khả năng chịu đựng tốt trước kịch bản xấu"
        elif abs(pd_change_pct) < 30:
            impact_level = "ẢNH HƯỞNG VỪA PHẢI 🟡"
            impact_desc = "Doanh nghiệp chịu ảnh hưởng đáng kể nhưng vẫn kiểm soát được"
        elif abs(pd_change_pct) < 50:
            impact_level = "ẢNH HƯỞNG LỚN 🟠"
            impact_desc = "Doanh nghiệp chịu tác động mạnh, cần có biện pháp phòng ngừa"
        else:
            impact_level = "ẢNH HƯỞNG RẤT LỚN 🔴"
            impact_desc = "Doanh nghiệp gặp rủi ro nghiêm trọng, cần hành động khẩn cấp"

        # Tạo chuỗi so sánh 14 chỉ số
        indicators_comparison = ""
        indicator_names = {
            'X_1': 'Biên LN gộp',
            'X_2': 'Biên LN trước thuế',
            'X_3': 'ROA',
            'X_4': 'ROE',
            'X_5': 'Nợ/TS',
            'X_6': 'Nợ/VCSH',
            'X_7': 'TT hiện hành (CR)',
            'X_8': 'TT nhanh',
            'X_9': 'Trả lãi',
            'X_10': 'Trả nợ gốc',
            'X_11': 'Tạo tiền/VCSH',
            'X_12': 'Vòng quay HTK',
            'X_13': 'Kỳ thu tiền',
            'X_14': 'Hiệu suất TS'
        }

        for key in ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14']:
            before = indicators_before.get(key, 0)
            after = indicators_after.get(key, 0)
            change = ((after - before) / before * 100) if before != 0 else 0
            arrow = "↓" if change < 0 else "↑" if change > 0 else "→"
            indicators_comparison += f"{key} ({indicator_names[key]}): {before:.4f} → {after:.4f} ({arrow}{abs(change):.1f}%)\n"

        # Lấy thông tin kịch bản
        scenario_name = scenario_info.get('name', 'N/A')
        changes = scenario_info.get('changes', {})
        revenue_change = changes.get('revenue', 0)
        interest_change = changes.get('interest', 0)
        roe_change = changes.get('roe', 0)
        cr_change = changes.get('cr', 0)

        prompt = f"""
Bạn là chuyên gia phân tích rủi ro tín dụng cao cấp của Agribank, chuyên về stress testing và mô phỏng kịch bản.

Dựa trên kết quả mô phỏng kịch bản kinh tế xấu, hãy phân tích chuyên sâu và đưa ra khuyến nghị chiến lược.

**KỊCH BẢN ĐÃ ÁP DỤNG:**
- Tên kịch bản: {scenario_name}
- Doanh thu thuần: {revenue_change:+.0f}%
- Chi phí lãi vay: {interest_change:+.0f}%
- ROE: {roe_change:+.0f}%
- Current Ratio (CR): {cr_change:+.0f}%

**KẾT QUẢ MÔ PHỎNG:**
- PD trước khi áp kịch bản: {pd_before:.2f}%
- PD sau khi áp kịch bản: {pd_after:.2f}%
- Thay đổi PD: {pd_change_pct:+.2f}%
- Mức độ ảnh hưởng: {impact_level}

**SO SÁNH 14 CHỈ SỐ TÀI CHÍNH (TRƯỚC → SAU):**
{indicators_comparison}

**YÊU CẦU PHÂN TÍCH:**

Hãy phân tích theo cấu trúc sau (bằng tiếng Việt, chuyên nghiệp, tối đa 600 từ):

1. **Đánh giá Tổng quan:**
   - Đánh giá khả năng chịu đựng của doanh nghiệp trước kịch bản {scenario_name}
   - Phân tích mức độ nghiêm trọng của thay đổi PD ({pd_change_pct:+.2f}%)
   - So sánh mức độ rủi ro trước và sau khi áp kịch bản

2. **Phân tích Chi tiết Tác động:**
   - Chỉ số nào bị ảnh hưởng NHIỀU NHẤT (thay đổi > 10%)?
   - Chỉ số nào vẫn ổn định (thay đổi < 5%)?
   - Phân tích chuỗi tác động: Doanh thu giảm → Lợi nhuận giảm → Khả năng trả nợ giảm
   - Đánh giá khả năng thanh toán (X_7, X_8, X_9, X_10) sau kịch bản

3. **Đánh giá Độ Bền Vững:**
   - Doanh nghiệp có thể tồn tại được bao lâu trong kịch bản này?
   - Điểm mạnh nào giúp doanh nghiệp chống đỡ?
   - Điểm yếu nào khiến doanh nghiệp dễ bị tổn thương?

4. **KHUYẾN NGHỊ CHIẾN LƯỢC** (QUAN TRỌNG):
   - **Đối với Ngân hàng:**
     * Có nên tiếp tục cho vay doanh nghiệp này trong điều kiện khủng hoảng?
     * Nếu CÓ: Đề xuất hạn mức, lãi suất, thời hạn, tài sản đảm bảo
     * Nếu KHÔNG: Giải thích rõ lý do
     * Biện pháp giảm thiểu rủi ro (covenant, giám sát chặt chẽ, v.v.)

   - **Đối với Doanh nghiệp:**
     * Cần chuẩn bị gì để đối phó với kịch bản xấu?
     * Ưu tiên cải thiện chỉ số nào?
     * Chiến lược tài chính nên điều chỉnh như thế nào?

5. **Kết luận:**
   - Tổng kết ngắn gọn về khả năng phục hồi của doanh nghiệp
   - Đánh giá cuối cùng về mức độ rủi ro tín dụng

Hãy trình bày rõ ràng, có cấu trúc, tập trung vào insight chiến lược.
"""

        try:
            # Gọi Gemini API
            response = self.model.generate_content(prompt)
            result = response.text
            return result

        except Exception as e:
            return f"❌ Lỗi khi phân tích kịch bản: {str(e)}"

    def analyze_survival_results(self, data: Dict[str, Any]) -> str:
        """
        Phân tích kết quả Survival Analysis bằng Gemini AI

        Args:
            data: Dict chứa:
                - indicators: 14 chỉ số tài chính
                - median_time_to_default: Median time (tháng)
                - survival_probabilities: Prob tại 6/12/24 tháng
                - risk_classification: Thông tin phân loại rủi ro
                - hazard_ratios: Top 5 hazard ratios
                - survival_curve: Timeline và probabilities
                - warning: Cảnh báo (nếu có)

        Returns:
            Phân tích chi tiết từ Gemini
        """
        # Lấy dữ liệu
        indicators = data.get('indicators', {})
        median_time = data.get('median_time_to_default', 0)
        survival_probs = data.get('survival_probabilities', {})
        risk_info = data.get('risk_classification', {})
        hazard_ratios = data.get('hazard_ratios', [])
        warning = data.get('warning', None)

        # Tạo danh sách 14 chỉ số với tên tiếng Việt
        indicator_names = {
            'X_1': 'Biên lợi nhuận gộp',
            'X_2': 'Biên lợi nhuận trước thuế',
            'X_3': 'ROA',
            'X_4': 'ROE',
            'X_5': 'Hệ số nợ trên tài sản',
            'X_6': 'Hệ số nợ trên VCSH',
            'X_7': 'Khả năng thanh toán hiện hành',
            'X_8': 'Khả năng thanh toán nhanh',
            'X_9': 'Khả năng trả lãi',
            'X_10': 'Khả năng trả nợ gốc',
            'X_11': 'Khả năng tạo tiền/VCSH',
            'X_12': 'Vòng quay hàng tồn kho',
            'X_13': 'Kỳ thu tiền bình quân',
            'X_14': 'Hiệu suất sử dụng tài sản'
        }

        # Format 14 chỉ số
        indicators_text = ""
        for key in sorted(indicators.keys()):
            if key in indicator_names:
                indicators_text += f"- {key} ({indicator_names[key]}): {indicators[key]:.4f}\n"

        # Format hazard ratios
        hazard_text = ""
        for i, hr in enumerate(hazard_ratios, 1):
            feature_name = hr.get('feature_name', 'N/A')
            ratio = hr.get('hazard_ratio', 1.0)
            significance = hr.get('significance', 'N/A')

            # Giải thích hazard ratio
            if ratio > 1:
                interpretation = f"tăng rủi ro {(ratio - 1) * 100:.1f}%"
            elif ratio < 1:
                interpretation = f"giảm rủi ro {(1 - ratio) * 100:.1f}%"
            else:
                interpretation = "không ảnh hưởng"

            hazard_text += f"{i}. {feature_name}: HR = {ratio:.3f} ({interpretation}) - {significance}\n"

        # Format survival probabilities
        survival_text = ""
        for time, prob in sorted(survival_probs.items()):
            default_prob = (1 - prob) * 100
            survival_text += f"- Tại tháng {int(time)}: Xác suất sống sót {prob * 100:.1f}%, Xác suất vỡ nợ {default_prob:.1f}%\n"

        # Phân loại cấp độ rủi ro
        risk_level = risk_info.get('level', 'N/A')
        risk_description = risk_info.get('description', 'N/A')

        # Cảnh báo
        warning_text = ""
        if warning:
            warning_text = f"\n⚠️ **CẢNH BÁO:** {warning.get('message', '')}\n"
            warning_text += f"**Khuyến nghị:** {warning.get('recommendation', '')}\n"

        prompt = f"""
Bạn là chuyên gia phân tích rủi ro tín dụng cao cấp của Agribank với 20 năm kinh nghiệm về Survival Analysis và Time-to-Default modeling.

Dựa trên kết quả phân tích sống sót (Survival Analysis) của doanh nghiệp, hãy đưa ra phân tích chuyên sâu và khuyến nghị chiến lược cho Agribank.

**CHỈ SỐ TÀI CHÍNH DOANH NGHIỆP:**
{indicators_text}

**KẾT QUẢ SURVIVAL ANALYSIS:**

📊 **Median Time-to-Default:** {median_time:.1f} tháng
   - Doanh nghiệp có 50% xác suất vỡ nợ trong vòng {median_time:.1f} tháng tới

📈 **Xác suất Sống sót & Vỡ nợ theo Thời gian:**
{survival_text}

🎯 **Phân loại Rủi ro:** {risk_level}
   - {risk_description}

🔬 **Top 5 Yếu tố Rủi ro Quan trọng (Hazard Ratios):**
{hazard_text}

**GHI CHÚ VỀ HAZARD RATIO:**
- HR > 1: Chỉ số này làm TĂNG nguy cơ vỡ nợ (càng cao càng nguy hiểm)
- HR < 1: Chỉ số này làm GIẢM nguy cơ vỡ nợ (bảo vệ doanh nghiệp)
- HR = 1: Chỉ số không ảnh hưởng đến rủi ro
{warning_text}

**YÊU CẦU PHÂN TÍCH:**

Hãy phân tích theo cấu trúc sau (bằng tiếng Việt chuyên nghiệp, 500-700 từ):

### 1. ĐÁNH GIÁ TỔNG QUAN VỀ KHẢ NĂNG SỐNG SÓT
- Đánh giá median time-to-default {median_time:.1f} tháng có ý nghĩa gì?
- So sánh với các mốc thời gian quan trọng: 6 tháng, 12 tháng, 24 tháng
- Xác suất vỡ nợ tại các thời điểm quan trọng cao hay thấp?
- Nhận xét về xu hướng survival curve (giảm nhanh hay giảm chậm?)

### 2. PHÂN TÍCH CÁC YẾU TỐ RỦI RO QUAN TRỌNG
- Phân tích TOP 5 chỉ số có Hazard Ratio cao nhất
- Chỉ số nào đang "kéo doanh nghiệp xuống vực thẳm" (HR >> 1)?
- Chỉ số nào đang "bảo vệ doanh nghiệp" (HR << 1)?
- Giải thích cơ chế tác động của các chỉ số này

### 3. SO SÁNH VỚI CHUẨN MỰC NGÀNH
- Median time {median_time:.1f} tháng là ngắn hay dài so với doanh nghiệp cùng ngành?
- Xác suất sống sót tại 12 tháng là {survival_probs.get(12, 0) * 100:.1f}% - đánh giá cao hay thấp?
- Doanh nghiệp này thuộc nhóm nào: Xuất sắc / Khỏe mạnh / Trung bình / Yếu / Nguy cấp?

### 4. KHUYẾN NGHỊ CHO AGRIBANK (QUAN TRỌNG)

**A. Quyết định Tín dụng:**
- ✅ **CÓ NÊN CHO VAY?** Giải thích rõ lý do
- 💰 **Hạn mức tối đa:** Đề xuất cụ thể (VD: 5 tỷ, 10 tỷ, 50 tỷ...)
- 📅 **Thời hạn vay:** Ngắn hạn (<6 tháng) / Trung hạn (6-12 tháng) / Dài hạn (>12 tháng)?
- 🏦 **Lãi suất:** Thấp hơn thị trường / Bằng thị trường / Cao hơn thị trường (risk premium)?
- 🏠 **Tài sản đảm bảo:** Có yêu cầu không? Tỷ lệ bao nhiêu (70%, 100%, 120%...)?

**B. Biện pháp Quản lý Rủi ro:**
- 📋 **Covenant (Điều khoản ràng buộc):** Chỉ số nào cần theo dõi?
- 🔍 **Giám sát:** Định kỳ hàng tháng / quý / năm?
- 🚨 **Early Warning Signals:** Dấu hiệu cảnh báo sớm nào cần chú ý?
- 🛡️ **Biện pháp dự phòng:** Chuẩn bị gì nếu doanh nghiệp xấu đi?

### 5. KHUYẾN NGHỊ CHO DOANH NGHIỆP
- Chỉ số nào cần CẢI THIỆN KHẨN CẤP để kéo dài thời gian sống sót?
- Đề xuất lộ trình hành động cụ thể (3-6 tháng tới)
- Chiến lược tài chính nên điều chỉnh như thế nào?

### 6. KẾT LUẬN & RATING ĐỀ XUẤT
- Tóm tắt ngắn gọn (2-3 câu)
- Đề xuất xếp hạng tín dụng: AAA / AA / A / BBB / BB / B / CCC / CC / C / D
- Quyết định cuối cùng: ✅ PHÊ DUYỆT / ⚠️ PHÊ DUYỆT CÓ ĐIỀU KIỆN / ❌ TỪ CHỐI

**QUAN TRỌNG:** Phân tích phải cụ thể, dựa trên số liệu thực tế, tập trung vào insight và hành động cụ thể, KHÔNG chung chung.
"""

        try:
            # Gọi Gemini API
            response = self.model.generate_content(prompt)
            result = response.text
            return result

        except Exception as e:
            return f"❌ Lỗi khi phân tích survival results: {str(e)}"


# Khởi tạo instance global
gemini_analyzer = None


def get_gemini_analyzer(api_key: str = None) -> GeminiAnalyzer:
    """
    Lấy instance của GeminiAnalyzer (singleton pattern)

    Args:
        api_key: API key của Gemini

    Returns:
        GeminiAnalyzer instance
    """
    global gemini_analyzer
    if gemini_analyzer is None:
        gemini_analyzer = GeminiAnalyzer(api_key)
    return gemini_analyzer
