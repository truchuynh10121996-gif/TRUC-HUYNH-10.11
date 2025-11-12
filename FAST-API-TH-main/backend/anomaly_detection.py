"""
Module Anomaly Detection System - Hệ thống Phát hiện Bất thường
Sử dụng Isolation Forest để phát hiện doanh nghiệp có hành vi bất thường
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os


class AnomalyDetectionSystem:
    """
    Hệ thống Phát hiện Bất thường (Anomaly Detection System)

    Chức năng chính:
    1. Train Isolation Forest model trên DN khỏe mạnh (label=0)
    2. Tính Anomaly Score (0-100) cho DN mới
    3. Phát hiện các features bất thường (so với P5, P95)
    4. Phân loại loại bất thường (Point/Contextual/Collective)
    5. Tạo giải thích bằng Gemini AI
    """

    def __init__(self):
        """Khởi tạo Anomaly Detection System"""
        self.model = None
        self.scaler = StandardScaler()
        self.thresholds = {}  # P5, P25, P50, P75, P95 cho 14 features
        self.feature_names = []
        self.healthy_stats = {}  # Thống kê DN khỏe mạnh

        # Tên đầy đủ của 14 chỉ số
        self.indicator_names = {
            'X_1': 'Biên lợi nhuận gộp',
            'X_2': 'Biên lợi nhuận trước thuế',
            'X_3': 'ROA (Lợi nhuận/Tài sản)',
            'X_4': 'ROE (Lợi nhuận/VCSH)',
            'X_5': 'Nợ/Tài sản',
            'X_6': 'Nợ/Vốn chủ sở hữu',
            'X_7': 'Thanh toán hiện hành',
            'X_8': 'Thanh toán nhanh',
            'X_9': 'Khả năng trả lãi',
            'X_10': 'Khả năng trả nợ gốc',
            'X_11': 'Tạo tiền/VCSH',
            'X_12': 'Vòng quay hàng tồn kho',
            'X_13': 'Kỳ thu tiền bình quân',
            'X_14': 'Hiệu suất sử dụng tài sản'
        }

    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Isolation Forest model trên DN khỏe mạnh

        Args:
            df: DataFrame chứa 14 chỉ số (X_1 → X_14) + cột 'label' (0=khỏe mạnh, 1=vỡ nợ)

        Returns:
            Dict chứa:
            - feature_statistics: Thống kê 14 features (P5, P25, P50, P75, P95)
            - contamination_rate: Tỷ lệ contamination
        """
        print("🔄 Bắt đầu train Anomaly Detection System...")

        # 1. LỌC DN KHỎE MẠNH (label == 0)
        healthy_df = df[df['label'] == 0].copy()
        print(f"✅ Có {len(healthy_df)} DN khỏe mạnh để train")

        # 2. CHUẨN BỊ FEATURES
        self.feature_names = [f'X_{i}' for i in range(1, 15)]
        X_healthy = healthy_df[self.feature_names].values

        # 3. CHUẨN HÓA DỮ LIỆU (FIT TRÊN DN KHỎE MẠNH)
        X_scaled = self.scaler.fit_transform(X_healthy)

        # 4. TÍNH THRESHOLDS (P5, P25, P50, P75, P95) CHO 14 FEATURES
        percentiles = [5, 25, 50, 75, 95]
        for i, feature in enumerate(self.feature_names):
            self.thresholds[feature] = {
                f'P{p}': np.percentile(X_healthy[:, i], p) for p in percentiles
            }

        # 5. TÍNH THỐNG KÊ DN KHỎE MẠNH (để so sánh)
        for i, feature in enumerate(self.feature_names):
            self.healthy_stats[feature] = {
                'mean': np.mean(X_healthy[:, i]),
                'std': np.std(X_healthy[:, i]),
                'min': np.min(X_healthy[:, i]),
                'max': np.max(X_healthy[:, i])
            }

        # 6. TRAIN ISOLATION FOREST
        print("📊 Training Isolation Forest...")
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # 5% DN bất thường
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        print("✅ Train Isolation Forest hoàn tất!")

        # 7. CHUẨN BỊ KẾT QUẢ TRẢ VỀ
        feature_statistics = []
        for feature in self.feature_names:
            feature_statistics.append({
                'feature': feature,
                'name': self.indicator_names[feature],
                'P5': round(self.thresholds[feature]['P5'], 4),
                'P25': round(self.thresholds[feature]['P25'], 4),
                'P50': round(self.thresholds[feature]['P50'], 4),
                'P75': round(self.thresholds[feature]['P75'], 4),
                'P95': round(self.thresholds[feature]['P95'], 4),
                'mean': round(self.healthy_stats[feature]['mean'], 4)
            })

        return {
            'feature_statistics': feature_statistics,
            'contamination_rate': 0.05,
            'num_healthy_samples': len(healthy_df),
            'num_total_samples': len(df)
        }

    def calculate_anomaly_score(self, indicators: Dict[str, float]) -> float:
        """
        Tính Anomaly Score (0-100) cho DN mới

        Args:
            indicators: Dict chứa 14 chỉ số (X_1 → X_14)

        Returns:
            anomaly_score: Điểm bất thường (0-100), càng cao càng bất thường
        """
        if self.model is None:
            raise ValueError("Model chưa được train. Vui lòng train model trước.")

        # Chuẩn bị input
        X_new = [[indicators[f] for f in self.feature_names]]
        X_scaled = self.scaler.transform(X_new)

        # Tính decision_function (raw score)
        # decision_function: càng âm càng bất thường, càng dương càng bình thường
        raw_score = self.model.decision_function(X_scaled)[0]

        # Normalize về [0, 100]
        # Dựa trên kinh nghiệm: decision_function thường trong khoảng [-0.5, 0.5]
        # -0.5 → 100 (rất bất thường)
        # 0.5 → 0 (rất bình thường)
        anomaly_score = max(0, min(100, (0.5 - raw_score) * 100))

        return round(anomaly_score, 2)

    def detect_abnormal_features(self, indicators: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Phát hiện các features bất thường (so với P5, P95)

        Args:
            indicators: Dict chứa 14 chỉ số (X_1 → X_14)

        Returns:
            List of Dict chứa thông tin các features bất thường:
            [{
                'feature_name': str,
                'current_value': float,
                'p5': float,
                'p50': float,
                'p95': float,
                'deviation_percent': float,
                'severity': str  # 'high' hoặc 'medium'
            }]
        """
        abnormal_features = []

        for feature in self.feature_names:
            current_value = indicators[feature]
            p5 = self.thresholds[feature]['P5']
            p50 = self.thresholds[feature]['P50']
            p95 = self.thresholds[feature]['P95']

            # Kiểm tra bất thường
            is_abnormal = False
            deviation_percent = 0
            severity = 'medium'

            if current_value < p5:
                # Thấp hơn P5 → Bất thường
                is_abnormal = True
                deviation_percent = ((p5 - current_value) / abs(p5) * 100) if p5 != 0 else 0
                severity = 'high' if deviation_percent > 50 else 'medium'
            elif current_value > p95:
                # Cao hơn P95 → Bất thường
                is_abnormal = True
                deviation_percent = ((current_value - p95) / abs(p95) * 100) if p95 != 0 else 0

                # ĐẶC BIỆT: Đối với một số chỉ số, cao hơn P95 là TỐT (không phải bất thường)
                good_if_high = ['X_1', 'X_2', 'X_3', 'X_4', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_14']
                if feature in good_if_high:
                    # Chỉ số này cao là tốt → Không coi là bất thường
                    is_abnormal = False
                else:
                    # Chỉ số này cao là xấu (X_5, X_6, X_13)
                    severity = 'high' if deviation_percent > 50 else 'medium'

            if is_abnormal:
                abnormal_features.append({
                    'feature_code': feature,
                    'feature_name': self.indicator_names[feature],
                    'current_value': round(current_value, 4),
                    'p5': round(p5, 4),
                    'p50': round(p50, 4),
                    'p95': round(p95, 4),
                    'deviation_percent': round(abs(deviation_percent), 2),
                    'severity': severity,
                    'direction': 'low' if current_value < p5 else 'high'
                })

        # Sắp xếp theo độ lệch giảm dần
        abnormal_features.sort(key=lambda x: x['deviation_percent'], reverse=True)

        return abnormal_features

    def classify_anomaly_type(self, indicators: Dict[str, float], abnormal_features: List[Dict]) -> str:
        """
        Phân loại loại bất thường

        Args:
            indicators: Dict chứa 14 chỉ số
            abnormal_features: List các features bất thường

        Returns:
            anomaly_type: "Point Anomaly", "Contextual Anomaly", hoặc "Collective Anomaly"
        """
        num_abnormal = len(abnormal_features)

        if num_abnormal == 0:
            return "Normal"
        elif num_abnormal == 1:
            return "Point Anomaly"
        elif num_abnormal >= 5:
            return "Collective Anomaly"
        else:
            return "Contextual Anomaly"

    def generate_gemini_explanation(
        self,
        indicators: Dict[str, float],
        anomaly_score: float,
        abnormal_features: List[Dict],
        anomaly_type: str,
        gemini_api_key: str
    ) -> str:
        """
        Tạo giải thích văn xuôi bằng Gemini AI

        Args:
            indicators: Dict chứa 14 chỉ số
            anomaly_score: Điểm bất thường (0-100)
            abnormal_features: List các features bất thường
            anomaly_type: Loại bất thường
            gemini_api_key: Gemini API key

        Returns:
            explanation: Giải thích văn xuôi (tiếng Việt, 200-300 từ)
        """
        try:
            import google.generativeai as genai

            # Cấu hình Gemini API
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            # Tạo prompt chi tiết
            prompt = f"""
Bạn là chuyên gia phát hiện gian lận tài chính của Agribank. Hãy phân tích kết quả phát hiện bất thường dưới đây.

**THÔNG TIN DOANH NGHIỆP:**

**Anomaly Score:** {anomaly_score}/100 (Điểm bất thường)
**Loại bất thường:** {anomaly_type}

**14 CHỈ SỐ TÀI CHÍNH:**
"""

            # Thêm 14 chỉ số
            for feature in self.feature_names:
                prompt += f"- {self.indicator_names[feature]} ({feature}): {indicators[feature]:.4f}\n"

            prompt += f"\n**CÁC CHỈ SỐ BẤT THƯỜNG ({len(abnormal_features)} chỉ số):**\n"

            if len(abnormal_features) > 0:
                for ab in abnormal_features[:5]:  # Top 5
                    prompt += f"""
- {ab['feature_name']} ({ab['feature_code']}):
  + Giá trị hiện tại: {ab['current_value']:.4f}
  + Ngưỡng bình thường: P5={ab['p5']:.4f}, P50={ab['p50']:.4f}, P95={ab['p95']:.4f}
  + Độ lệch: {ab['deviation_percent']:.2f}% ({ab['direction']})
  + Mức độ nghiêm trọng: {ab['severity']}
"""
            else:
                prompt += "- Không có chỉ số bất thường\n"

            prompt += """

**YÊU CẦU PHÂN TÍCH:**

Hãy viết báo cáo phân tích chi tiết (200-300 từ, tiếng Việt) với cấu trúc sau:

## 🔍 ĐÁNH GIÁ TỔNG QUAN
(2-3 câu mô tả mức độ bất thường của doanh nghiệp)

## 📊 PHÂN TÍCH CÁC CHỈ SỐ BẤT THƯỜNG
(Phân tích chi tiết từng chỉ số bất thường, giải thích tại sao bất thường)

## ⚠️ RỦI RO TIỀM ẨN
(Liệt kê 2-3 rủi ro có thể xảy ra: gian lận, báo cáo sai, hoạt động bất thường, v.v.)

## 💡 KHUYẾN NGHỊ
(Đưa ra 2-3 khuyến nghị cụ thể cho ngân hàng: cần xem xét, kiểm tra sâu, yêu cầu giải trình, v.v.)

---
**Lưu ý:** Viết ngắn gọn, chuyên nghiệp, dễ hiểu. Tập trung vào phát hiện dấu hiệu bất thường và đưa ra cảnh báo cụ thể.
"""

            # Gọi Gemini API
            response = model.generate_content(prompt)
            explanation = response.text

            return explanation

        except Exception as e:
            return f"Lỗi khi gọi Gemini API: {str(e)}"


# Khởi tạo singleton instance
anomaly_system = AnomalyDetectionSystem()
