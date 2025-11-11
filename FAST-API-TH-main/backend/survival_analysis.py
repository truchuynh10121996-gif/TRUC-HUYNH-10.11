"""
Survival Analysis System for Credit Risk Assessment
Implements Cox Proportional Hazards, Random Survival Forest, and Kaplan-Meier Estimator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Warning: lifelines not installed. Install with: pip install lifelines")

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("Warning: scikit-survival not installed. Install with: pip install scikit-survival")


class SurvivalAnalysisSystem:
    """
    Hệ thống phân tích sống sót cho đánh giá rủi ro tín dụng
    Dự báo thời gian đến khi doanh nghiệp vỡ nợ (Time-to-Default)
    """

    def __init__(self):
        self.cox_model = None
        self.rsf_model = None
        self.km_fitter = None
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
        self.training_data = None
        self.metrics = {}

    def prepare_data(self, df: pd.DataFrame, duration_col: str = 'months_to_default',
                    event_col: str = 'event') -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Chuẩn bị dữ liệu cho survival analysis

        Args:
            df: DataFrame chứa 14 chỉ số tài chính + months_to_default + event
            duration_col: Tên cột thời gian (tháng)
            event_col: Tên cột event (1=vỡ nợ, 0=censored)

        Returns:
            X: Features DataFrame
            durations: Array thời gian
            events: Array events
        """
        # Lấy 14 chỉ số tài chính
        X = df[self.feature_names].copy()

        # Xử lý missing values
        X = X.fillna(X.median())

        # Lấy duration và event
        durations = df[duration_col].values
        events = df[event_col].values if event_col in df.columns else np.ones(len(df))

        # Đảm bảo duration > 0
        durations = np.maximum(durations, 0.1)

        return X, durations, events

    def train_cox_model(self, df: pd.DataFrame, duration_col: str = 'months_to_default',
                       event_col: str = 'event') -> Dict[str, Any]:
        """
        Huấn luyện Cox Proportional Hazards Model

        Args:
            df: Training data với 14 chỉ số + months_to_default + event

        Returns:
            Dict chứa metrics và model info
        """
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines package is required. Install with: pip install lifelines")

        # Chuẩn bị dữ liệu
        X, durations, events = self.prepare_data(df, duration_col, event_col)

        # Tạo DataFrame cho Cox model
        cox_data = X.copy()
        cox_data['duration'] = durations
        cox_data['event'] = events

        # Huấn luyện Cox model
        self.cox_model = CoxPHFitter(penalizer=0.01)
        self.cox_model.fit(cox_data, duration_col='duration', event_col='event')

        # Tính C-index (concordance index)
        c_index = self.cox_model.concordance_index_

        # Lưu training data để dùng cho Kaplan-Meier baseline
        self.training_data = cox_data

        # Lưu metrics
        self.metrics['cox_c_index'] = float(c_index)
        self.metrics['cox_log_likelihood'] = float(self.cox_model.log_likelihood_)

        return {
            'model_type': 'Cox Proportional Hazards',
            'c_index': float(c_index),
            'log_likelihood': float(self.cox_model.log_likelihood_),
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(self.feature_names)
        }

    def train_random_survival_forest(self, df: pd.DataFrame,
                                     duration_col: str = 'months_to_default',
                                     event_col: str = 'event',
                                     n_estimators: int = 100) -> Dict[str, Any]:
        """
        Huấn luyện Random Survival Forest

        Args:
            df: Training data
            n_estimators: Số lượng trees

        Returns:
            Dict chứa metrics
        """
        if not SKSURV_AVAILABLE:
            raise ImportError("scikit-survival required. Install with: pip install scikit-survival")

        # Chuẩn bị dữ liệu
        X, durations, events = self.prepare_data(df, duration_col, event_col)

        # Tạo structured array cho scikit-survival
        y = Surv.from_arrays(event=events.astype(bool), time=durations)

        # Huấn luyện RSF
        self.rsf_model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )
        self.rsf_model.fit(X, y)

        # Tính C-index
        c_index = self.rsf_model.score(X, y)

        # Lưu metrics
        self.metrics['rsf_c_index'] = float(c_index)
        self.metrics['rsf_n_estimators'] = n_estimators

        return {
            'model_type': 'Random Survival Forest',
            'c_index': float(c_index),
            'n_estimators': n_estimators,
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(self.feature_names)
        }

    def calculate_kaplan_meier(self, df: pd.DataFrame = None,
                              duration_col: str = 'months_to_default',
                              event_col: str = 'event') -> Dict[str, Any]:
        """
        Tính Kaplan-Meier Estimator (baseline survival function)

        Args:
            df: Data (nếu None, dùng training data)

        Returns:
            Dict với survival function và timeline
        """
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines package required")

        # Sử dụng training data nếu không có df
        if df is None:
            if self.training_data is None:
                raise ValueError("No training data available. Train Cox model first.")
            durations = self.training_data['duration'].values
            events = self.training_data['event'].values
        else:
            _, durations, events = self.prepare_data(df, duration_col, event_col)

        # Fit Kaplan-Meier
        self.km_fitter = KaplanMeierFitter()
        self.km_fitter.fit(durations, events)

        # Lấy survival function
        timeline = self.km_fitter.survival_function_.index.tolist()
        survival_probs = self.km_fitter.survival_function_['KM_estimate'].tolist()

        # Tính median survival time
        median_survival = self.km_fitter.median_survival_time_

        return {
            'timeline': timeline,
            'survival_probabilities': survival_probs,
            'median_survival_time': float(median_survival) if not np.isnan(median_survival) else None,
            'event_count': int(events.sum()),
            'censored_count': int((1 - events).sum())
        }

    def predict_survival_curve(self, indicators: Dict[str, float],
                               model_type: str = 'cox',
                               timeline: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Dự báo survival curve cho một doanh nghiệp mới

        Args:
            indicators: Dict với 14 chỉ số tài chính (X_1 đến X_14)
            model_type: 'cox' hoặc 'rsf'
            timeline: List các thời điểm (tháng) để dự báo

        Returns:
            Dict với survival probabilities tại các thời điểm
        """
        # Tạo DataFrame từ indicators
        X_new = pd.DataFrame([indicators])[self.feature_names]

        # Xử lý missing values
        X_new = X_new.fillna(0)

        if model_type == 'cox':
            if self.cox_model is None:
                raise ValueError("Cox model not trained. Call train_cox_model() first.")

            # Dự báo survival function
            surv_func = self.cox_model.predict_survival_function(X_new)

            # Lấy survival curve của sample đầu tiên (cột đầu tiên, không phải row đầu tiên)
            surv_curve = surv_func.iloc[:, 0]  # Series với index = timeline

            # Nếu không có timeline, dùng timeline từ model
            if timeline is None:
                timeline = surv_curve.index.tolist()
                survival_probs = [float(p) for p in surv_curve.values.tolist()]
            else:
                # Lấy survival probabilities tại các thời điểm cụ thể
                survival_probs = []
                for t in timeline:
                    if t in surv_curve.index:
                        survival_probs.append(float(surv_curve.loc[t]))
                    else:
                        # Interpolate nếu thời điểm không có trong index
                        idx = np.searchsorted(surv_curve.index, t)
                        if idx == 0:
                            survival_probs.append(float(surv_curve.iloc[0]))
                        elif idx >= len(surv_curve):
                            survival_probs.append(float(surv_curve.iloc[-1]))
                        else:
                            # Linear interpolation
                            t1, p1 = surv_curve.index[idx-1], surv_curve.iloc[idx-1]
                            t2, p2 = surv_curve.index[idx], surv_curve.iloc[idx]
                            prob = p1 + (t - t1) * (p2 - p1) / (t2 - t1)
                            survival_probs.append(float(prob))

        elif model_type == 'rsf':
            if self.rsf_model is None:
                raise ValueError("RSF model not trained. Call train_random_survival_forest() first.")

            # Dự báo survival function
            surv_funcs = self.rsf_model.predict_survival_function(X_new, return_array=True)

            # Lấy survival probabilities của sample đầu tiên
            surv_probs_array = surv_funcs[0]  # Array survival probs của sample đầu tiên
            unique_times = self.rsf_model.unique_times_

            # Timeline từ RSF model
            if timeline is None:
                timeline = unique_times.tolist()
                survival_probs = [float(p) for p in surv_probs_array.tolist()]
            else:
                # Lấy survival probabilities tại các thời điểm cụ thể
                survival_probs = []
                for t in timeline:
                    idx = np.searchsorted(unique_times, t)
                    if idx == 0:
                        survival_probs.append(float(surv_probs_array[0]))
                    elif idx >= len(unique_times):
                        survival_probs.append(float(surv_probs_array[-1]))
                    elif t == unique_times[idx]:
                        survival_probs.append(float(surv_probs_array[idx]))
                    else:
                        # Linear interpolation
                        t1, p1 = unique_times[idx-1], surv_probs_array[idx-1]
                        t2, p2 = unique_times[idx], surv_probs_array[idx]
                        prob = p1 + (t - t1) * (p2 - p1) / (t2 - t1)
                        survival_probs.append(float(prob))
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'cox' or 'rsf'.")

        return {
            'timeline': timeline,
            'survival_probabilities': survival_probs,
            'model_type': model_type
        }

    def calculate_median_time_to_default(self, indicators: Dict[str, float],
                                         model_type: str = 'cox') -> float:
        """
        Tính median time-to-default cho một doanh nghiệp

        Args:
            indicators: Dict với 14 chỉ số
            model_type: 'cox' hoặc 'rsf'

        Returns:
            Median time (tháng)
        """
        # Dự báo survival curve
        result = self.predict_survival_curve(indicators, model_type)

        timeline = result['timeline']
        survival_probs = result['survival_probabilities']

        # Kiểm tra timeline và survival_probs có dữ liệu
        if not timeline or not survival_probs:
            raise ValueError("Timeline hoặc survival probabilities rỗng")

        # Tìm thời điểm mà survival probability = 0.5
        for i, prob in enumerate(survival_probs):
            if prob <= 0.5:
                if i == 0:
                    return float(timeline[0])
                else:
                    # Linear interpolation
                    t1, p1 = timeline[i-1], survival_probs[i-1]
                    t2, p2 = timeline[i], survival_probs[i]

                    # Tránh chia cho 0
                    if abs(p2 - p1) < 1e-10:
                        median_time = t1
                    else:
                        median_time = t1 + (0.5 - p1) * (t2 - t1) / (p2 - p1)
                    return float(median_time)

        # Nếu survival probability không bao giờ xuống dưới 0.5
        # Doanh nghiệp có rủi ro thấp, median time rất lớn
        return float(timeline[-1])  # Return max time

    def get_hazard_ratios(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Lấy hazard ratios từ Cox model (top K chỉ số quan trọng nhất)

        Args:
            top_k: Số lượng chỉ số muốn lấy

        Returns:
            List các dict với feature name, hazard ratio, và p-value
        """
        if self.cox_model is None:
            raise ValueError("Cox model not trained. Call train_cox_model() first.")

        # Lấy hazard ratios và p-values
        hazard_ratios = np.exp(self.cox_model.params_)  # exp(coef) = hazard ratio
        p_values = self.cox_model.summary['p']
        confidence_intervals = self.cox_model.confidence_intervals_

        # Tạo list kết quả
        results = []
        for feature in self.feature_names:
            if feature in hazard_ratios.index:
                results.append({
                    'feature_code': feature,
                    'feature_name': self.feature_name_mapping[feature],
                    'hazard_ratio': float(hazard_ratios[feature]),
                    'coefficient': float(self.cox_model.params_[feature]),
                    'p_value': float(p_values[feature]),
                    'ci_lower': float(confidence_intervals.loc[feature].iloc[0]),
                    'ci_upper': float(confidence_intervals.loc[feature].iloc[1]),
                    'significance': 'Có ý nghĩa' if p_values[feature] < 0.05 else 'Không có ý nghĩa'
                })

        # Sắp xếp theo absolute hazard ratio (càng xa 1.0 càng quan trọng)
        results.sort(key=lambda x: abs(np.log(x['hazard_ratio'])), reverse=True)

        return results[:top_k]

    def get_individual_risk_contributions(self, indicators: Dict[str, float],
                                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tính risk contribution của TỪNG CHỈ SỐ cho DOANH NGHIỆP CỤ THỂ này
        (KHÁC với get_hazard_ratios - trả về model-level metrics giống nhau cho mọi DN)

        Args:
            indicators: Dict với 14 chỉ số tài chính của doanh nghiệp CỤ THỂ
            top_k: Số lượng chỉ số muốn lấy

        Returns:
            List các dict với feature name, contribution, và diễn giải CỤ THỂ cho DN này

        Ví dụ:
            - DN A (ROA = 10%): X_3 contribution = -2.5 (giảm rủi ro)
            - DN B (ROA = -5%): X_3 contribution = +1.8 (tăng rủi ro)
        """
        if self.cox_model is None:
            raise ValueError("Cox model not trained. Call train_cox_model() first.")

        # Tạo DataFrame cho doanh nghiệp này
        company_data = pd.DataFrame([indicators])

        # Đảm bảo có đủ 14 chỉ số
        for feature in self.feature_names:
            if feature not in company_data.columns:
                company_data[feature] = 0

        # Sắp xếp theo thứ tự features
        company_data = company_data[self.feature_names]

        # Xử lý missing values
        company_data = company_data.fillna(0)

        # Lấy coefficients từ Cox model
        coefficients = self.cox_model.params_
        p_values = self.cox_model.summary['p']

        # Tính training data statistics (mean) để so sánh
        if self.training_data is not None:
            training_means = self.training_data[self.feature_names].mean()
            training_stds = self.training_data[self.feature_names].std()
        else:
            training_means = pd.Series(0, index=self.feature_names)
            training_stds = pd.Series(1, index=self.feature_names)

        # Tính risk contributions cho DOANH NGHIỆP NÀY
        results = []
        total_log_hazard = 0

        for feature in self.feature_names:
            if feature in coefficients.index:
                coef = float(coefficients[feature])
                company_value = float(company_data[feature].iloc[0])
                mean_value = float(training_means[feature])
                std_value = float(training_stds[feature])
                p_val = float(p_values[feature])

                # Risk contribution = coef × (value - mean)
                # Positive contribution = TĂNG rủi ro
                # Negative contribution = GIẢM rủi ro
                contribution = coef * (company_value - mean_value)
                total_log_hazard += contribution

                # Standardized contribution (so với độ lệch chuẩn)
                if std_value > 0:
                    z_score = (company_value - mean_value) / std_value
                    contribution_std = coef * z_score
                else:
                    z_score = 0
                    contribution_std = 0

                # Diễn giải
                if abs(contribution) < 0.01:
                    interpretation = "⚪ Không ảnh hưởng (gần trung bình)"
                elif contribution > 0:
                    # TĂNG rủi ro
                    if contribution > 1.0:
                        interpretation = f"🔴 TĂNG rủi ro MẠNH (+{contribution:.2f})"
                    elif contribution > 0.5:
                        interpretation = f"🟠 TĂNG rủi ro TRUNG BÌNH (+{contribution:.2f})"
                    else:
                        interpretation = f"🟡 Tăng rủi ro nhẹ (+{contribution:.2f})"
                else:
                    # GIẢM rủi ro
                    if contribution < -1.0:
                        interpretation = f"🟢 GIẢM rủi ro MẠNH ({contribution:.2f})"
                    elif contribution < -0.5:
                        interpretation = f"🟢 GIẢM rủi ro TRUNG BÌNH ({contribution:.2f})"
                    else:
                        interpretation = f"🟢 Giảm rủi ro nhẹ ({contribution:.2f})"

                # So sánh với trung bình
                if company_value > mean_value:
                    comparison = f"CAO hơn TB {abs(company_value - mean_value):.3f}"
                elif company_value < mean_value:
                    comparison = f"THẤP hơn TB {abs(company_value - mean_value):.3f}"
                else:
                    comparison = "BẰNG trung bình"

                results.append({
                    'feature_code': feature,
                    'feature_name': self.feature_name_mapping[feature],

                    # Giá trị của DOANH NGHIỆP NÀY
                    'company_value': company_value,
                    'mean_value': mean_value,
                    'z_score': z_score,
                    'comparison': comparison,

                    # Risk contribution CỤ THỂ
                    'risk_contribution': contribution,
                    'risk_contribution_std': contribution_std,
                    'interpretation': interpretation,

                    # Model info (để tham khảo)
                    'coefficient': coef,
                    'p_value': p_val,
                    'is_significant': p_val < 0.05
                })

        # Sắp xếp theo absolute contribution (chỉ số ảnh hưởng mạnh nhất lên đầu)
        results.sort(key=lambda x: abs(x['risk_contribution']), reverse=True)

        # Thêm thông tin tổng hợp
        top_results = results[:top_k]

        # Tính % contribution so với tổng
        total_abs_contribution = sum(abs(r['risk_contribution']) for r in results)
        for r in top_results:
            if total_abs_contribution > 0:
                r['contribution_pct'] = abs(r['risk_contribution']) / total_abs_contribution * 100
            else:
                r['contribution_pct'] = 0

        return top_results

    def get_survival_probabilities_at_times(self, indicators: Dict[str, float],
                                           times: List[float] = [6, 12, 24],
                                           model_type: str = 'cox') -> Dict[float, float]:
        """
        Tính survival probability tại các thời điểm cụ thể

        Args:
            indicators: Dict với 14 chỉ số
            times: List các thời điểm (tháng)
            model_type: 'cox' hoặc 'rsf'

        Returns:
            Dict {time: survival_probability}
        """
        # Dự báo survival curve
        result = self.predict_survival_curve(indicators, model_type, timeline=None)

        timeline = np.array(result['timeline'])
        survival_probs = np.array(result['survival_probabilities'])

        # Interpolate để lấy survival prob tại các thời điểm cụ thể
        probs_at_times = {}
        for t in times:
            if t <= timeline[0]:
                probs_at_times[t] = float(survival_probs[0])
            elif t >= timeline[-1]:
                probs_at_times[t] = float(survival_probs[-1])
            else:
                # Linear interpolation
                idx = np.searchsorted(timeline, t)
                t1, p1 = timeline[idx-1], survival_probs[idx-1]
                t2, p2 = timeline[idx], survival_probs[idx]
                prob = p1 + (t - t1) * (p2 - p1) / (t2 - t1)
                probs_at_times[t] = float(prob)

        return probs_at_times

    def get_risk_classification(self, median_time: float) -> Dict[str, str]:
        """
        Phân loại mức độ rủi ro dựa trên median time-to-default

        Args:
            median_time: Median time (tháng)

        Returns:
            Dict với risk level và color
        """
        if median_time < 6:
            return {
                'level': 'Rất cao',
                'color': '#FFE8E8',
                'text_color': '#C62828',
                'icon': '🔴',
                'description': 'Nguy cơ vỡ nợ cực kỳ cao trong vòng 6 tháng'
            }
        elif median_time < 12:
            return {
                'level': 'Cao',
                'color': '#FFE0CC',
                'text_color': '#E65100',
                'icon': '🟠',
                'description': 'Nguy cơ vỡ nợ cao trong vòng 1 năm'
            }
        elif median_time < 24:
            return {
                'level': 'Trung bình',
                'color': '#FFF9E8',
                'text_color': '#F57C00',
                'icon': '🟡',
                'description': 'Cần theo dõi chặt chẽ trong vòng 2 năm'
            }
        elif median_time < 36:
            return {
                'level': 'Thấp',
                'color': '#E8FFF0',
                'text_color': '#1B5E20',
                'icon': '🟢',
                'description': 'Rủi ro thấp, tình trạng tài chính ổn định'
            }
        else:
            return {
                'level': 'Rất thấp',
                'color': '#C8F5DC',
                'text_color': '#0D5B2B',
                'icon': '🟢',
                'description': 'Tình trạng tài chính rất tốt, rủi ro rất thấp'
            }

    def save_models(self, filepath: str = 'survival_models.pkl'):
        """Lưu models"""
        models = {
            'cox_model': self.cox_model,
            'rsf_model': self.rsf_model,
            'km_fitter': self.km_fitter,
            'training_data': self.training_data,
            'metrics': self.metrics
        }
        joblib.dump(models, filepath)
        return {'status': 'success', 'filepath': filepath}

    def load_models(self, filepath: str = 'survival_models.pkl'):
        """Load models"""
        models = joblib.load(filepath)
        self.cox_model = models['cox_model']
        self.rsf_model = models['rsf_model']
        self.km_fitter = models['km_fitter']
        self.training_data = models['training_data']
        self.metrics = models['metrics']
        return {'status': 'success', 'metrics': self.metrics}


# Singleton instance
survival_system = SurvivalAnalysisSystem()
