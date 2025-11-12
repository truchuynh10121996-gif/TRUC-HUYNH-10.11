"""
Module Early Warning System - Hệ thống Cảnh báo Rủi ro Sớm
Sử dụng ML (Stacking + K-Means + Gemini AI) để chẩn đoán sức khỏe tài chính doanh nghiệp
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os


class EarlyWarningSystem:
    """
    Hệ thống Cảnh báo Rủi ro Sớm (Early Warning System)

    Chức năng chính:
    1. Train Stacking model (RF + XGB + GB) + K-Means clustering
    2. Tính Health Score (0-100) dựa trên 14 chỉ số và feature importances
    3. Phân loại mức rủi ro (Safe/Watch/Warning/Alert)
    4. Phát hiện điểm yếu (top 3 chỉ số xa ngưỡng an toàn nhất)
    5. Xác định vị trí trong cluster
    6. Dự báo PD trong tương lai (3/6/12 tháng) theo kịch bản vĩ mô
    7. Tạo báo cáo chẩn đoán bằng Gemini AI
    """

    def __init__(self):
        """Khởi tạo Early Warning System"""
        self.stacking_model = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.thresholds = {}  # Ngưỡng an toàn cho 14 chỉ số
        self.feature_importances = {}
        self.training_data = None
        self.cluster_info = {}

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

    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Stacking model và K-Means clustering

        Args:
            df: DataFrame chứa 14 chỉ số (X_1 → X_14) + cột 'label' (0=không vỡ nợ, 1=vỡ nợ)

        Returns:
            Dict chứa thông tin về training:
            - num_samples: Số lượng mẫu
            - feature_importances: Feature importances từ RandomForest
            - cluster_distribution: Phân bố các cluster
        """
        print("🔄 Bắt đầu train Early Warning System...")

        # Lưu training data
        self.training_data = df.copy()

        # Tách features và labels
        feature_cols = [f'X_{i}' for i in range(1, 15)]
        X = df[feature_cols].values
        y = df['label'].values

        # 1. TRAIN STACKING MODEL (RF + XGB + GB, meta=LogisticRegression)
        print("📊 Training Stacking Classifier...")

        # Base models
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )

        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        # Meta model
        meta_model = LogisticRegression(max_iter=1000)

        # Stacking
        self.stacking_model = StackingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('gb', gb_model)
            ],
            final_estimator=meta_model,
            cv=5
        )

        self.stacking_model.fit(X, y)
        print("✅ Stacking model trained!")

        # Extract feature importances từ RandomForest layer
        rf_estimator = self.stacking_model.named_estimators_['rf']
        importances = rf_estimator.feature_importances_

        self.feature_importances = {
            feature_cols[i]: float(importances[i])
            for i in range(len(feature_cols))
        }

        print("📈 Feature Importances:")
        for feature, importance in sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True):
            print(f"   {feature}: {importance:.4f}")

        # 2. TRAIN K-MEANS CLUSTERING (4 clusters)
        print("🔍 Training K-Means (4 clusters)...")

        # Chỉ cluster nhóm không vỡ nợ (label=0)
        X_healthy = df[df['label'] == 0][feature_cols].values

        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.kmeans.fit(X_healthy)

        cluster_labels = self.kmeans.predict(X_healthy)

        # Tính thông tin cluster
        for cluster_id in range(4):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = X_healthy[cluster_mask]

            self.cluster_info[cluster_id] = {
                'size': int(np.sum(cluster_mask)),
                'center': self.kmeans.cluster_centers_[cluster_id].tolist(),
                'avg_values': np.mean(cluster_data, axis=0).tolist()
            }

        print("✅ K-Means trained!")
        print(f"   Cluster sizes: {[self.cluster_info[i]['size'] for i in range(4)]}")

        # 3. TÍNH NGƯỠNG AN TOÀN (percentile P40, P50, P60 của nhóm label=0)
        print("📏 Calculating safety thresholds...")

        df_healthy = df[df['label'] == 0]

        for col in feature_cols:
            # Một số chỉ số càng cao càng tốt (sinh lời, thanh toán)
            # Một số chỉ số càng thấp càng tốt (nợ, kỳ thu tiền)

            # Chỉ số càng cao càng tốt: X_1, X_2, X_3, X_4, X_7, X_8, X_9, X_10, X_11, X_12, X_14
            # Chỉ số càng thấp càng tốt: X_5, X_6, X_13

            if col in ['X_5', 'X_6', 'X_13']:
                # Càng thấp càng tốt → ngưỡng an toàn là P60 (không vượt quá)
                self.thresholds[col] = {
                    'safe_zone': float(df_healthy[col].quantile(0.40)),
                    'watch_zone': float(df_healthy[col].quantile(0.50)),
                    'warning_zone': float(df_healthy[col].quantile(0.60)),
                    'direction': 'lower_is_better'
                }
            else:
                # Càng cao càng tốt → ngưỡng an toàn là P40 (không thấp hơn)
                self.thresholds[col] = {
                    'safe_zone': float(df_healthy[col].quantile(0.60)),
                    'watch_zone': float(df_healthy[col].quantile(0.50)),
                    'warning_zone': float(df_healthy[col].quantile(0.40)),
                    'direction': 'higher_is_better'
                }

        print("✅ Thresholds calculated!")

        # 4. Trả về thông tin training
        result = {
            'num_samples': len(df),
            'num_healthy': int(np.sum(df['label'] == 0)),
            'num_default': int(np.sum(df['label'] == 1)),
            'feature_importances': self.feature_importances,
            'cluster_distribution': {
                f'cluster_{i}': self.cluster_info[i]['size']
                for i in range(4)
            }
        }

        print("✅ Early Warning System trained successfully!")
        return result

    def calculate_health_score(self, indicators: Dict[str, float]) -> float:
        """
        Tính Health Score (0-100) dựa trên 60% PD + 40% Statistical

        Args:
            indicators: Dict chứa 14 chỉ số (X_1 → X_14)

        Returns:
            Health Score (0-100)

        Công thức:
            1. Tính Statistical Score dựa trên thresholds và feature importances
            2. Tính PD Score từ stacking_model
            3. Health Score = 60% * (100 - PD) + 40% * Statistical Score
        """
        if not self.feature_importances:
            raise ValueError("Model chưa được train. Vui lòng gọi train_models() trước.")

        if self.stacking_model is None:
            raise ValueError("Stacking model chưa được train. Vui lòng gọi train_models() trước.")

        # 1. TÍNH STATISTICAL SCORE (40%)
        total_score = 0.0
        total_weight = 0.0

        for indicator, value in indicators.items():
            if indicator not in self.thresholds:
                continue

            threshold_info = self.thresholds[indicator]
            importance = self.feature_importances.get(indicator, 0.0)

            # Normalize về [0, 1]
            if threshold_info['direction'] == 'higher_is_better':
                # Càng cao càng tốt
                safe = threshold_info['safe_zone']
                warning = threshold_info['warning_zone']

                if value >= safe:
                    normalized = 1.0
                elif value <= warning:
                    normalized = 0.0
                else:
                    normalized = (value - warning) / (safe - warning) if safe != warning else 0.5
            else:
                # Càng thấp càng tốt
                safe = threshold_info['safe_zone']
                warning = threshold_info['warning_zone']

                if value <= safe:
                    normalized = 1.0
                elif value >= warning:
                    normalized = 0.0
                else:
                    normalized = (warning - value) / (warning - safe) if warning != safe else 0.5

            # Weighted sum
            total_score += normalized * importance
            total_weight += importance

        # Statistical score (0-100)
        statistical_score = (total_score / total_weight * 100) if total_weight > 0 else 50.0
        statistical_score = max(0.0, min(100.0, statistical_score))

        # 2. TÍNH PD SCORE (60%)
        feature_cols = [f'X_{i}' for i in range(1, 15)]
        X_input = [[indicators[col] for col in feature_cols]]
        pd_value = self.stacking_model.predict_proba(X_input)[0, 1] * 100  # PD in %

        # PD Score: 100 - PD (PD càng thấp → score càng cao)
        pd_score = max(0.0, min(100.0, 100 - pd_value))

        # 3. KẾT HỢP: 60% PD + 40% Statistical
        health_score = 0.6 * pd_score + 0.4 * statistical_score

        # Giới hạn trong [0, 100]
        health_score = max(0.0, min(100.0, health_score))

        return round(health_score, 2)

    def classify_risk_level(self, health_score: float) -> Dict[str, str]:
        """
        Phân loại mức rủi ro dựa trên Health Score

        Args:
            health_score: Health Score (0-100)

        Returns:
            Dict chứa risk_level và risk_level_color
        """
        if health_score >= 80:
            return {
                'risk_level': 'Safe',
                'risk_level_color': '#10B981',
                'risk_level_icon': '🟢',
                'risk_level_text': 'An toàn'
            }
        elif health_score >= 60:
            return {
                'risk_level': 'Watch',
                'risk_level_color': '#F59E0B',
                'risk_level_icon': '🟡',
                'risk_level_text': 'Theo dõi'
            }
        elif health_score >= 40:
            return {
                'risk_level': 'Warning',
                'risk_level_color': '#FF8C00',
                'risk_level_icon': '🟠',
                'risk_level_text': 'Cảnh báo'
            }
        else:
            return {
                'risk_level': 'Alert',
                'risk_level_color': '#EF4444',
                'risk_level_icon': '🔴',
                'risk_level_text': 'Nguy hiểm'
            }

    def detect_weaknesses(self, indicators: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Phát hiện điểm yếu (top 3 chỉ số xa ngưỡng an toàn nhất)

        Args:
            indicators: Dict chứa 14 chỉ số

        Returns:
            List top 3 chỉ số yếu nhất
        """
        weaknesses = []

        for indicator, value in indicators.items():
            if indicator not in self.thresholds:
                continue

            threshold_info = self.thresholds[indicator]
            safe_threshold = threshold_info['safe_zone']
            direction = threshold_info['direction']

            # Tính gap (khoảng cách so với ngưỡng an toàn)
            if direction == 'higher_is_better':
                gap = value - safe_threshold
                severity = 'critical' if gap < -safe_threshold * 0.3 else 'moderate' if gap < 0 else 'low'
            else:
                gap = safe_threshold - value
                severity = 'critical' if gap < -safe_threshold * 0.3 else 'moderate' if gap < 0 else 'low'

            # Tính percentile
            if self.training_data is not None:
                healthy_data = self.training_data[self.training_data['label'] == 0]
                if indicator in healthy_data.columns:
                    percentile = (healthy_data[indicator] < value).sum() / len(healthy_data) * 100
                else:
                    percentile = 50.0
            else:
                percentile = 50.0

            weaknesses.append({
                'indicator': indicator,
                'name': self.indicator_names.get(indicator, indicator),
                'current_value': round(value, 4),
                'safe_threshold': round(safe_threshold, 4),
                'gap': round(gap, 4),
                'percentile': round(percentile, 1),
                'severity': severity,
                'direction': direction
            })

        # Sắp xếp theo gap (âm nhất = yếu nhất)
        weaknesses.sort(key=lambda x: x['gap'])

        # Trả về top 3
        return weaknesses[:3]

    def get_cluster_position(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """
        Xác định vị trí DN trong cluster

        Args:
            indicators: Dict chứa 14 chỉ số

        Returns:
            Dict chứa cluster_id, cluster_name, position_percentile, cluster_avg_pd
        """
        if self.kmeans is None:
            raise ValueError("K-Means chưa được train. Vui lòng gọi train_models() trước.")

        # Chuẩn bị input
        feature_cols = [f'X_{i}' for i in range(1, 15)]
        X_input = np.array([[indicators[col] for col in feature_cols]])

        # Predict cluster
        cluster_id = int(self.kmeans.predict(X_input)[0])

        # Tính percentile trong toàn dataset
        if self.training_data is not None:
            healthy_data = self.training_data[self.training_data['label'] == 0]

            # Tính khoảng cách đến center của cluster
            center = self.kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(healthy_data[feature_cols].values - center, axis=1)
            current_distance = np.linalg.norm(X_input[0] - center)

            # Percentile: vị trí của DN trong toàn bộ healthy dataset
            # Tính dựa trên health score
            health_scores = []
            for _, row in healthy_data.iterrows():
                row_indicators = {col: row[col] for col in feature_cols}
                hs = self.calculate_health_score(row_indicators)
                health_scores.append(hs)

            current_health_score = self.calculate_health_score(indicators)
            position_percentile = (np.array(health_scores) < current_health_score).sum() / len(health_scores) * 100
        else:
            position_percentile = 50.0

        # Tên cluster (dựa trên percentile)
        if position_percentile >= 75:
            cluster_name = "🟢 Nhóm A - Xuất sắc"
        elif position_percentile >= 50:
            cluster_name = "🟡 Nhóm B - Tốt"
        elif position_percentile >= 25:
            cluster_name = "🟠 Nhóm C - Yếu"
        else:
            cluster_name = "🔴 Nhóm D - Rất yếu"

        # Tính cluster avg PD (nếu có stacking model)
        cluster_avg_pd = 0.0
        if self.stacking_model is not None and self.training_data is not None:
            # Lấy tất cả DN trong cluster này
            cluster_mask = self.kmeans.predict(healthy_data[feature_cols].values) == cluster_id
            cluster_data = healthy_data[cluster_mask]

            if len(cluster_data) > 0:
                # Dự báo PD cho cluster
                X_cluster = cluster_data[feature_cols].values
                cluster_pds = self.stacking_model.predict_proba(X_cluster)[:, 1] * 100
                cluster_avg_pd = float(np.mean(cluster_pds))

        # Tính median indicators của cluster
        if self.training_data is not None:
            healthy_data = self.training_data[self.training_data['label'] == 0]
            cluster_mask = self.kmeans.predict(healthy_data[feature_cols].values) == cluster_id
            cluster_data = healthy_data[cluster_mask]

            cluster_median_indicators = {}
            for col in feature_cols:
                if col in cluster_data.columns:
                    cluster_median_indicators[col] = float(cluster_data[col].median())
                else:
                    cluster_median_indicators[col] = 0.0
        else:
            cluster_median_indicators = {col: 0.0 for col in feature_cols}

        return {
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'position_percentile': round(position_percentile, 1),
            'cluster_avg_pd': round(cluster_avg_pd, 2),
            'cluster_median_indicators': cluster_median_indicators
        }

    def project_future_pd(
        self,
        indicators: Dict[str, float],
        months: int,
        scenario: str,
        excel_processor,
        industry_code: str = "manufacturing"
    ) -> float:
        """
        Dự báo PD trong tương lai theo kịch bản vĩ mô

        Args:
            indicators: Dict 14 chỉ số hiện tại
            months: Số tháng dự báo (3/6/12)
            scenario: Kịch bản ("recession_mild", "recession_moderate", "crisis")
            excel_processor: Instance của ExcelProcessor
            industry_code: Mã ngành

        Returns:
            PD dự báo (%)
        """
        if self.stacking_model is None:
            raise ValueError("Stacking model chưa được train. Vui lòng gọi train_models() trước.")

        # Kịch bản vĩ mô
        macro_scenarios = {
            'recession_mild': {
                'gdp_growth_pct': -1.5,
                'inflation_cpi_pct': 6.0,
                'inflation_ppi_pct': 8.0,
                'policy_rate_change_bps': 100,
                'fx_usd_vnd_pct': 3.0
            },
            'recession_moderate': {
                'gdp_growth_pct': -3.5,
                'inflation_cpi_pct': 10.0,
                'inflation_ppi_pct': 14.0,
                'policy_rate_change_bps': 200,
                'fx_usd_vnd_pct': 6.0
            },
            'crisis': {
                'gdp_growth_pct': -6.0,
                'inflation_cpi_pct': 15.0,
                'inflation_ppi_pct': 20.0,
                'policy_rate_change_bps': 300,
                'fx_usd_vnd_pct': 10.0
            }
        }

        if scenario not in macro_scenarios:
            scenario = 'recession_mild'

        macro_vars = macro_scenarios[scenario]

        # Kênh truyền dẫn macro → micro
        micro_shocks = excel_processor.macro_to_micro_transmission(
            gdp_growth_pct=macro_vars['gdp_growth_pct'],
            inflation_cpi_pct=macro_vars['inflation_cpi_pct'],
            inflation_ppi_pct=macro_vars['inflation_ppi_pct'],
            policy_rate_change_bps=macro_vars['policy_rate_change_bps'],
            fx_usd_vnd_pct=macro_vars['fx_usd_vnd_pct'],
            industry_code=industry_code
        )

        # Điều chỉnh mức độ shock theo số tháng (càng xa càng mạnh)
        time_multiplier = months / 12  # 3 tháng = 0.25, 6 tháng = 0.5, 12 tháng = 1.0

        # Tính 14 chỉ số sau shock
        indicators_after = excel_processor.simulate_scenario_full_propagation(
            original_indicators=indicators,
            revenue_change_pct=micro_shocks['revenue_change_pct'] * time_multiplier,
            interest_rate_change_pct=micro_shocks['interest_rate_change_pct'] * time_multiplier,
            cogs_change_pct=micro_shocks['cogs_change_pct'] * time_multiplier,
            liquidity_shock_pct=micro_shocks['liquidity_shock_pct'] * time_multiplier
        )

        # Dự báo PD
        feature_cols = [f'X_{i}' for i in range(1, 15)]
        X_future = np.array([[indicators_after[col] for col in feature_cols]])

        pd_future = self.stacking_model.predict_proba(X_future)[0, 1] * 100

        return round(pd_future, 2)

    def generate_gemini_diagnosis(
        self,
        health_score: float,
        risk_info: Dict[str, str],
        weaknesses: List[Dict[str, Any]],
        cluster_info: Dict[str, Any],
        pd_projections: Dict[str, Any],
        current_pd: float,
        gemini_api_key: Optional[str] = None
    ) -> str:
        """
        Tạo báo cáo chẩn đoán bằng Gemini AI

        Args:
            health_score: Health Score
            risk_info: Thông tin risk level
            weaknesses: Top 3 điểm yếu
            cluster_info: Thông tin cluster
            pd_projections: Dự báo PD tương lai
            current_pd: PD hiện tại
            gemini_api_key: Gemini API key

        Returns:
            Báo cáo chẩn đoán (tiếng Việt)
        """
        # Lấy Gemini API key từ environment nếu không được truyền vào
        if gemini_api_key is None:
            gemini_api_key = os.getenv('GEMINI_API_KEY')

        if not gemini_api_key:
            return self._generate_fallback_diagnosis(
                health_score, risk_info, weaknesses, cluster_info, pd_projections, current_pd
            )

        try:
            import google.generativeai as genai

            # Configure Gemini
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            # Tạo prompt
            prompt = f"""
Bạn là chuyên gia phân tích tín dụng của Agribank. Hãy viết báo cáo chẩn đoán sức khỏe tài chính cho doanh nghiệp này.

**THÔNG TIN CHẨN ĐOÁN:**

1. **Health Score:** {health_score:.2f}/100
2. **Mức rủi ro:** {risk_info['risk_level_icon']} {risk_info['risk_level_text']}
3. **PD hiện tại:** {current_pd:.2f}%
4. **Vị trí:** {cluster_info['cluster_name']} (Xếp hạng {cluster_info['position_percentile']:.1f}% trong 1300 DN)

**TOP 3 ĐIỂM YẾU:**
{chr(10).join([f"- **{w['name']}**: Giá trị hiện tại {w['current_value']:.2f}, ngưỡng an toàn {w['safe_threshold']:.2f} (Gap: {w['gap']:.2f}, Mức độ: {w['severity']})" for w in weaknesses])}

**DỰ BÁO PD TƯƠNG LAI:**
- **3 tháng:**
  - Suy thoái nhẹ: {pd_projections.get('recession_mild', {}).get('3_months', 0):.2f}%
  - Suy thoái trung bình: {pd_projections.get('recession_moderate', {}).get('3_months', 0):.2f}%
  - Khủng hoảng: {pd_projections.get('crisis', {}).get('3_months', 0):.2f}%

- **6 tháng:**
  - Suy thoái nhẹ: {pd_projections.get('recession_mild', {}).get('6_months', 0):.2f}%
  - Suy thoái trung bình: {pd_projections.get('recession_moderate', {}).get('6_months', 0):.2f}%
  - Khủng hoảng: {pd_projections.get('crisis', {}).get('6_months', 0):.2f}%

- **12 tháng:**
  - Suy thoái nhẹ: {pd_projections.get('recession_mild', {}).get('12_months', 0):.2f}%
  - Suy thoái trung bình: {pd_projections.get('recession_moderate', {}).get('12_months', 0):.2f}%
  - Khủng hoảng: {pd_projections.get('crisis', {}).get('12_months', 0):.2f}%

**YÊU CẦU:**
Hãy viết báo cáo chẩn đoán với cấu trúc sau (sử dụng Markdown):

## 📋 CHẨN ĐOÁN TỔNG QUAN
(2-3 câu tóm tắt tình hình sức khỏe tài chính và mức độ rủi ro)

## 🔍 PHÂN TÍCH CHI TIẾT

### Điểm mạnh
(Liệt kê 2-3 điểm mạnh của DN)

### Điểm yếu cần cải thiện
(Phân tích chi tiết 3 điểm yếu được liệt kê ở trên, giải thích tại sao chúng quan trọng và ảnh hưởng như thế nào)

## 💡 KHUYẾN NGHỊ

### Ngắn hạn (0-3 tháng)
(2-3 khuyến nghị cụ thể)

### Trung hạn (3-12 tháng)
(2-3 khuyến nghị cụ thể)

## ⚠️ ĐÁNH GIÁ RỦI RO

### Khả năng chống chịu với suy thoái kinh tế
(Phân tích dựa trên dự báo PD tương lai)

### Quyết định tín dụng
(Khuyến nghị: Chấp thuận / Cân nhắc / Từ chối - kèm điều kiện cụ thể)

---
**Lưu ý:** Viết ngắn gọn, chuyên nghiệp, dễ hiểu. Tránh lặp lại thông tin. Tập trung vào insights và actionable recommendations.
"""

            # Gọi Gemini API
            response = model.generate_content(prompt)
            diagnosis = response.text

            return diagnosis

        except Exception as e:
            print(f"⚠️ Lỗi khi gọi Gemini API: {str(e)}")
            return self._generate_fallback_diagnosis(
                health_score, risk_info, weaknesses, cluster_info, pd_projections, current_pd
            )

    def _generate_fallback_diagnosis(
        self,
        health_score: float,
        risk_info: Dict[str, str],
        weaknesses: List[Dict[str, Any]],
        cluster_info: Dict[str, Any],
        pd_projections: Dict[str, Any],
        current_pd: float
    ) -> str:
        """Tạo báo cáo chẩn đoán fallback (không dùng Gemini)"""

        diagnosis = f"""## 📋 CHẨN ĐOÁN TỔNG QUAN

Doanh nghiệp có **Health Score {health_score:.2f}/100**, thuộc mức **{risk_info['risk_level_icon']} {risk_info['risk_level_text']}** với PD hiện tại **{current_pd:.2f}%**.

Vị trí: **{cluster_info['cluster_name']}** (xếp hạng **{cluster_info['position_percentile']:.1f}%** trong 1300 DN).

## 🔍 PHÂN TÍCH CHI TIẾT

### Điểm yếu cần cải thiện

"""

        # Liệt kê 3 điểm yếu
        for i, w in enumerate(weaknesses, 1):
            diagnosis += f"""{i}. **{w['name']}**
   - Giá trị hiện tại: {w['current_value']:.2f}
   - Ngưỡng an toàn: {w['safe_threshold']:.2f}
   - Khoảng cách: {w['gap']:.2f} ({w['severity']})
   - Percentile: {w['percentile']:.1f}%

"""

        diagnosis += f"""## 💡 KHUYẾN NGHỊ

### Ngắn hạn (0-3 tháng)
- Tập trung cải thiện các chỉ số yếu nhất (đặc biệt là {weaknesses[0]['name']})
- Tăng cường thanh khoản và quản lý dòng tiền
- Xem xét tái cơ cấu nợ nếu cần thiết

### Trung hạn (3-12 tháng)
- Cải thiện hiệu quả hoạt động kinh doanh
- Tối ưu hóa cấu trúc vốn
- Đa dạng hóa nguồn thu

## ⚠️ ĐÁNH GIÁ RỦI RO

### Khả năng chống chịu với suy thoái kinh tế

Dựa trên mô phỏng:
- **Suy thoái nhẹ (12 tháng):** PD tăng lên {pd_projections.get('recession_mild', {}).get('12_months', 0):.2f}%
- **Suy thoái trung bình (12 tháng):** PD tăng lên {pd_projections.get('recession_moderate', {}).get('12_months', 0):.2f}%
- **Khủng hoảng (12 tháng):** PD tăng lên {pd_projections.get('crisis', {}).get('12_months', 0):.2f}%

### Quyết định tín dụng

"""

        if health_score >= 70:
            diagnosis += "**✅ Khuyến nghị: Chấp thuận** - Doanh nghiệp có sức khỏe tài chính tốt."
        elif health_score >= 50:
            diagnosis += "**⚠️ Khuyến nghị: Cân nhắc** - Yêu cầu thêm tài sản đảm bảo hoặc điều kiện bổ sung."
        else:
            diagnosis += "**❌ Khuyến nghị: Từ chối hoặc yêu cầu cải thiện** - Rủi ro cao, cần cải thiện sức khỏe tài chính trước khi xem xét."

        return diagnosis


# Khởi tạo instance global
early_warning_system = EarlyWarningSystem()
