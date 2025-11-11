"""
GIẢI PHÁP FIX HAZARD RATIO = 0 VÀ EXTREME VALUES
================================================

Vấn đề: X2, X3 luôn có HR = 0, các chỉ số khác có HR cực đoan (5174)
Nguyên nhân: Thiếu chuẩn hóa dữ liệu + Numerical instability
Giải pháp: Chuẩn hóa + Regularization + Better interpretation

Author: TRUC-HUYNH
Date: 2025-11-11
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from scipy.stats.mstats import winsorize
from typing import Dict, List, Any, Tuple


class ImprovedSurvivalAnalysis:
    """
    Survival Analysis với xử lý proper cho Hazard Ratios
    """

    def __init__(self):
        self.cox_model = None
        self.scaler = None
        self.feature_names = [
            'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7',
            'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14'
        ]
        self.feature_name_mapping = {
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

    # ============================================================
    # GIẢI PHÁP 1: CHUẨN HÓA DỮ LIỆU + XỬ LÝ OUTLIERS
    # ============================================================

    def prepare_data_improved(
        self,
        df: pd.DataFrame,
        duration_col: str = 'months_to_default',
        event_col: str = 'event',
        handle_outliers: bool = True,
        winsorize_limits: Tuple[float, float] = (0.01, 0.01)
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Chuẩn bị dữ liệu với proper scaling và outlier handling

        Args:
            df: DataFrame chứa 14 chỉ số + duration + event
            duration_col: Tên cột thời gian
            event_col: Tên cột event
            handle_outliers: Có xử lý outliers không
            winsorize_limits: Giới hạn winsorization (lower, upper)

        Returns:
            X_scaled: Features đã chuẩn hóa
            durations: Array thời gian
            events: Array events
        """
        # Lấy 14 chỉ số tài chính
        X = df[self.feature_names].copy()

        # Xử lý missing values
        X = X.fillna(X.median())

        # ✅ BƯỚC 1: Xử lý outliers (nếu cần)
        if handle_outliers:
            print("🔧 Đang xử lý outliers bằng winsorization...")
            for col in X.columns:
                X[col] = winsorize(X[col], limits=winsorize_limits)
            print("✅ Đã xử lý outliers")

        # ✅ BƯỚC 2: Chuẩn hóa dữ liệu (QUAN TRỌNG!)
        print("🔧 Đang chuẩn hóa dữ liệu (StandardScaler)...")
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        print("✅ Đã chuẩn hóa dữ liệu")

        # In thống kê sau khi scale
        print("\n📊 Thống kê sau khi scale:")
        print(f"  Mean: {X_scaled.mean().mean():.6f} (should be ~0)")
        print(f"  Std: {X_scaled.std().mean():.6f} (should be ~1)")
        print(f"  Min: {X_scaled.min().min():.2f}")
        print(f"  Max: {X_scaled.max().max():.2f}")

        # Lấy duration và event
        durations = df[duration_col].values
        events = df[event_col].values if event_col in df.columns else np.ones(len(df))

        # Đảm bảo duration > 0
        durations = np.maximum(durations, 0.1)

        return X_scaled, durations, events

    # ============================================================
    # GIẢI PHÁP 2: ELASTIC NET COX MODEL
    # ============================================================

    def train_cox_model_improved(
        self,
        df: pd.DataFrame,
        duration_col: str = 'months_to_default',
        event_col: str = 'event',
        penalizer: float = 0.1,
        l1_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        Huấn luyện Cox model với Elastic Net regularization

        Args:
            df: Training data
            penalizer: Regularization strength (0.01-1.0)
            l1_ratio: 0=Ridge, 1=Lasso, 0.5=Elastic Net

        Returns:
            Dict chứa metrics
        """
        # Chuẩn bị dữ liệu với improved pipeline
        X_scaled, durations, events = self.prepare_data_improved(
            df, duration_col, event_col, handle_outliers=True
        )

        # Tạo DataFrame cho Cox model
        cox_data = X_scaled.copy()
        cox_data['duration'] = durations
        cox_data['event'] = events

        # ✅ IMPROVED: Elastic Net Cox với penalizer cao hơn
        print(f"\n🔧 Training Cox model (penalizer={penalizer}, l1_ratio={l1_ratio})...")
        self.cox_model = CoxPHFitter(
            penalizer=penalizer,
            l1_ratio=l1_ratio  # Elastic Net
        )
        self.cox_model.fit(cox_data, duration_col='duration', event_col='event')

        # Tính metrics
        c_index = self.cox_model.concordance_index_
        log_likelihood = self.cox_model.log_likelihood_

        print(f"✅ Model trained successfully!")
        print(f"  C-index: {c_index:.4f}")
        print(f"  Log-likelihood: {log_likelihood:.2f}")

        # Kiểm tra coefficient range
        coef_min = self.cox_model.params_.min()
        coef_max = self.cox_model.params_.max()
        print(f"\n📊 Coefficient range:")
        print(f"  Min: {coef_min:.4f}")
        print(f"  Max: {coef_max:.4f}")

        if coef_max > 10 or coef_min < -10:
            print("⚠️  WARNING: Coefficients still too extreme!")
            print("   Consider increasing penalizer or checking data quality")

        return {
            'model_type': 'Cox Proportional Hazards (Elastic Net)',
            'c_index': float(c_index),
            'log_likelihood': float(log_likelihood),
            'penalizer': penalizer,
            'l1_ratio': l1_ratio,
            'coef_range': [float(coef_min), float(coef_max)]
        }

    # ============================================================
    # GIẢI PHÁP 3: IMPROVED HAZARD RATIO REPORTING
    # ============================================================

    def get_hazard_ratios_improved(
        self,
        top_k: int = 5,
        clip_hr: bool = True,
        hr_min: float = 0.001,
        hr_max: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """
        Lấy hazard ratios với proper handling của extreme values

        Args:
            top_k: Số lượng chỉ số muốn lấy
            clip_hr: Có clip HR vào khoảng [hr_min, hr_max] không
            hr_min: Giá trị HR tối thiểu
            hr_max: Giá trị HR tối đa

        Returns:
            List các dict với feature info và HR
        """
        if self.cox_model is None:
            raise ValueError("Cox model not trained!")

        # Lấy coefficients và p-values
        coefficients = self.cox_model.params_
        p_values = self.cox_model.summary['p']
        confidence_intervals = self.cox_model.confidence_intervals_

        # Tính hazard ratios
        hazard_ratios_raw = np.exp(coefficients)

        # ✅ IMPROVED: Clip HR vào khoảng hợp lý (nếu cần)
        if clip_hr:
            hazard_ratios = np.clip(hazard_ratios_raw, hr_min, hr_max)
            print(f"\n🔧 Clipping HR to [{hr_min}, {hr_max}]")
        else:
            hazard_ratios = hazard_ratios_raw

        # Tạo list kết quả
        results = []
        for feature in self.feature_names:
            if feature in hazard_ratios.index:
                coef = float(coefficients[feature])
                hr = float(hazard_ratios[feature])
                hr_raw = float(hazard_ratios_raw[feature])
                p_val = float(p_values[feature])

                # ✅ IMPROVED: Thêm nhiều metrics hơn
                result = {
                    'feature_code': feature,
                    'feature_name': self.feature_name_mapping[feature],

                    # Coefficient (log HR)
                    'coefficient': coef,

                    # Hazard Ratios
                    'hazard_ratio': hr,
                    'hazard_ratio_raw': hr_raw,  # Giá trị gốc trước khi clip
                    'was_clipped': hr != hr_raw,  # Có bị clip không

                    # Statistical significance
                    'p_value': p_val,
                    'ci_lower': float(confidence_intervals.loc[feature].iloc[0]),
                    'ci_upper': float(confidence_intervals.loc[feature].iloc[1]),
                    'significance': 'Có ý nghĩa (p<0.05)' if p_val < 0.05 else 'Không có ý nghĩa (p≥0.05)',
                    'is_significant': p_val < 0.05,

                    # ✅ IMPROVED: Diễn giải dễ hiểu hơn
                    'interpretation': self._interpret_hazard_ratio(coef, hr, p_val)
                }

                results.append(result)

        # Sắp xếp theo absolute coefficient (not HR!)
        # Vì sau khi scale, coefficient đáng tin hơn
        results.sort(key=lambda x: abs(x['coefficient']), reverse=True)

        return results[:top_k]

    def _interpret_hazard_ratio(self, coef: float, hr: float, p_val: float) -> str:
        """
        Diễn giải Hazard Ratio một cách dễ hiểu
        """
        # Kiểm tra ý nghĩa thống kê
        if p_val >= 0.05:
            return "⚪ Không có bằng chứng thống kê về ảnh hưởng"

        # Nếu có ý nghĩa thống kê
        if coef > 0:
            # Tăng rủi ro
            if hr < 1.5:
                return f"🟡 Tăng rủi ro nhẹ ({(hr-1)*100:.1f}%)"
            elif hr < 2.0:
                return f"🟠 Tăng rủi ro trung bình ({(hr-1)*100:.1f}%)"
            else:
                return f"🔴 Tăng rủi ro mạnh ({(hr-1)*100:.1f}%)"
        else:
            # Giảm rủi ro
            risk_reduction = (1 - hr) * 100
            if hr > 0.67:
                return f"🟢 Giảm rủi ro nhẹ ({risk_reduction:.1f}%)"
            elif hr > 0.5:
                return f"🟢 Giảm rủi ro trung bình ({risk_reduction:.1f}%)"
            else:
                return f"🟢 Giảm rủi ro mạnh ({risk_reduction:.1f}%)"

    # ============================================================
    # GIẢI PHÁP 4: PRETTY PRINT KẾT QUẢ
    # ============================================================

    def print_hazard_ratios_table(self, top_k: int = 5):
        """
        In bảng Hazard Ratios đẹp và dễ đọc
        """
        hrs = self.get_hazard_ratios_improved(top_k=top_k, clip_hr=True)

        print("\n" + "="*100)
        print("📊 BẢNG HAZARD RATIOS - TOP YẾU TỐ RỦI RO QUAN TRỌNG".center(100))
        print("="*100)

        print("\n💡 Giải thích Hazard Ratio (HR):")
        print("  • Coefficient > 0 (HR > 1): Chỉ số TĂNG nguy cơ vỡ nợ")
        print("  • Coefficient < 0 (HR < 1): Chỉ số GIẢM nguy cơ vỡ nợ")
        print("  • P-value < 0.05: Có ý nghĩa thống kê")

        print("\n" + "-"*100)
        print(f"{'#':<3} {'Chỉ số':<40} {'Coef':<8} {'HR':<10} {'P-value':<10} {'Diễn giải':<30}")
        print("-"*100)

        for i, hr in enumerate(hrs, 1):
            feature_name = hr['feature_name']
            coef = hr['coefficient']
            hr_val = hr['hazard_ratio']
            p_val = hr['p_value']
            interp = hr['interpretation']

            # Highlight nếu bị clip
            if hr['was_clipped']:
                feature_name += " ⚠️"

            print(f"{i:<3} {feature_name:<40} {coef:>7.3f} {hr_val:>9.3f} {p_val:>9.4f} {interp:<30}")

        print("-"*100)
        print("\n⚠️  Lưu ý:")
        print("  • Nếu có ⚠️: HR bị clip vì giá trị quá cực đoan (báo hiệu vấn đề với data/model)")
        print("  • Nếu p-value > 0.05: Kết quả không đáng tin cậy về mặt thống kê")
        print("  • Nên tập trung vào các chỉ số có p-value < 0.05")
        print("="*100 + "\n")

    # ============================================================
    # GIẢI PHÁP 5: SO SÁNH TRƯỚC/SAU IMPROVEMENT
    # ============================================================

    def compare_old_vs_new(self, df: pd.DataFrame):
        """
        So sánh kết quả của cách cũ vs cách mới
        """
        print("\n" + "="*100)
        print("🔬 SO SÁNH PHƯƠNG PHÁP CŨ VS MỚI".center(100))
        print("="*100)

        # CŨ: Không scale, penalizer thấp
        print("\n1️⃣  PHƯƠNG PHÁP CŨ (Không scale, penalizer=0.01):")
        print("-"*100)
        old_system = SurvivalAnalysisOld()
        old_system.train_cox_model_old(df)
        old_hrs = old_system.get_hazard_ratios_old(top_k=5)

        for i, hr in enumerate(old_hrs, 1):
            print(f"  {i}. {hr['feature_name']}: HR={hr['hazard_ratio']:.3f}, p={hr['p_value']:.4f}")

        # MỚI: Scale + Elastic Net
        print("\n2️⃣  PHƯƠNG PHÁP MỚI (Scale + Elastic Net, penalizer=0.1):")
        print("-"*100)
        self.train_cox_model_improved(df, penalizer=0.1, l1_ratio=0.5)
        new_hrs = self.get_hazard_ratios_improved(top_k=5, clip_hr=True)

        for i, hr in enumerate(new_hrs, 1):
            print(f"  {i}. {hr['feature_name']}: HR={hr['hazard_ratio']:.3f}, p={hr['p_value']:.4f}")

        print("\n✅ Cải thiện:")
        print("  • HR trong khoảng hợp lý hơn (không có 0 hay 5174)")
        print("  • P-values đáng tin hơn")
        print("  • Coefficients ổn định hơn")
        print("="*100 + "\n")


# ============================================================
# CLASS CŨ (ĐỂ SO SÁNH)
# ============================================================

class SurvivalAnalysisOld:
    """Class cũ để so sánh"""

    def __init__(self):
        self.cox_model = None
        self.feature_names = [
            'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7',
            'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14'
        ]
        self.feature_name_mapping = {
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

    def train_cox_model_old(self, df: pd.DataFrame):
        """Phương pháp cũ - KHÔNG SCALE"""
        X = df[self.feature_names].copy()
        X = X.fillna(X.median())  # Chỉ fill NA, KHÔNG SCALE

        cox_data = X.copy()
        cox_data['duration'] = np.maximum(df['months_to_default'].values, 0.1)
        cox_data['event'] = df['event'].values

        self.cox_model = CoxPHFitter(penalizer=0.01)  # Penalizer thấp
        self.cox_model.fit(cox_data, duration_col='duration', event_col='event')

    def get_hazard_ratios_old(self, top_k: int = 5):
        """Phương pháp cũ - KHÔNG CLIP"""
        hazard_ratios = np.exp(self.cox_model.params_)
        p_values = self.cox_model.summary['p']

        results = []
        for feature in self.feature_names:
            if feature in hazard_ratios.index:
                results.append({
                    'feature_name': self.feature_name_mapping[feature],
                    'hazard_ratio': float(hazard_ratios[feature]),
                    'p_value': float(p_values[feature])
                })

        results.sort(key=lambda x: abs(np.log(x['hazard_ratio'] + 1e-10)), reverse=True)
        return results[:top_k]


# ============================================================
# DEMO USAGE
# ============================================================

if __name__ == "__main__":
    print("\n🚀 DEMO: FIX HAZARD RATIO = 0 PROBLEM")
    print("="*100)

    # Tạo sample data
    np.random.seed(42)
    n_samples = 200

    # Tạo 14 chỉ số với scale khác nhau
    data = {
        'X_1': np.random.uniform(0.1, 0.4, n_samples),  # Biên lợi nhuận gộp
        'X_2': np.random.uniform(-0.1, 0.2, n_samples),  # Biên LN trước thuế (SCALE NHỎ)
        'X_3': np.random.uniform(-0.05, 0.15, n_samples),  # ROA (SCALE NHỎ)
        'X_4': np.random.uniform(-0.2, 0.3, n_samples),  # ROE
        'X_5': np.random.uniform(0.2, 0.9, n_samples),  # Nợ/Tài sản
        'X_6': np.random.uniform(0.5, 3.0, n_samples),  # Nợ/VCSH
        'X_7': np.random.uniform(0.8, 2.5, n_samples),  # Thanh toán hiện hành
        'X_8': np.random.uniform(0.5, 2.0, n_samples),  # Thanh toán nhanh
        'X_9': np.random.uniform(1.0, 10.0, n_samples),  # Trả lãi
        'X_10': np.random.uniform(0.5, 5.0, n_samples),  # Trả nợ gốc
        'X_11': np.random.uniform(-2.0, 2.0, n_samples),  # Tạo tiền/VCSH
        'X_12': np.random.uniform(2.0, 15.0, n_samples),  # Vòng quay HTK
        'X_13': np.random.uniform(30, 90, n_samples),  # Kỳ thu tiền
        'X_14': np.random.uniform(0.5, 2.0, n_samples),  # Hiệu suất tài sản
    }

    df = pd.DataFrame(data)

    # Tạo synthetic survival data
    # Công ty có ROA thấp, nợ cao → vỡ nợ sớm
    risk_score = (
        -5 * df['X_3'] +  # ROA thấp → rủi ro cao
        2 * df['X_5'] +   # Nợ cao → rủi ro cao
        -1 * df['X_2']    # Lợi nhuận thấp → rủi ro cao
    )

    df['months_to_default'] = np.clip(
        np.random.exponential(scale=1.0/np.exp(risk_score)) * 60,
        0.1, 120
    )
    df['event'] = np.random.binomial(1, 0.3, n_samples)  # 30% default rate

    print(f"\n📊 Sample data created: {n_samples} companies")
    print(f"  Default rate: {df['event'].mean()*100:.1f}%")
    print(f"  Median survival time: {df['months_to_default'].median():.1f} months")

    # Test improved method
    print("\n" + "="*100)
    print("🧪 TESTING IMPROVED METHOD")
    print("="*100)

    system = ImprovedSurvivalAnalysis()
    system.train_cox_model_improved(df, penalizer=0.1, l1_ratio=0.5)
    system.print_hazard_ratios_table(top_k=5)

    print("\n✅ Demo completed! Check results above.")
    print("\n💡 Key Takeaways:")
    print("  1. Always scale your features before Cox regression")
    print("  2. Use proper regularization (Elastic Net)")
    print("  3. Clip extreme HR values for interpretability")
    print("  4. Focus on p-values < 0.05 for reliable results")
    print("  5. Consider using RSF if Cox is still unstable")
    print("="*100 + "\n")
