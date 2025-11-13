"""
FastAPI Backend - Hệ thống Đánh giá Rủi ro Tín dụng
Endpoints: /train, /predict, /predict-from-xlsx, /analyze, /export-report
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from dotenv import load_dotenv
import os

load_dotenv()  # Tải các biến môi trường từ file .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import pandas as pd
import os
import tempfile
from datetime import datetime
from model import credit_model
from gemini_api import get_gemini_analyzer
from excel_processor import excel_processor
from report_generator import ReportGenerator
from early_warning import early_warning_system
from anomaly_detection import anomaly_system
from survival_analysis import survival_system

# Khởi tạo FastAPI app
app = FastAPI(
    title="Credit Risk Assessment API",
    description="API đánh giá rủi ro tín dụng sử dụng Stacking Classifier",
    version="1.0.0"
)

# Cấu hình CORS để frontend Vue có thể gọi API
# Development: cho phép localhost:3000 (frontend Vue)
# Production: thay đổi origins theo domain thật
origins = [
    "http://localhost:3000",      # Vue dev server
    "http://localhost:5173",      # Vite alternative port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    # Thêm domain production khi deploy:
    # "https://yourdomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ================================================================================================
# HELPER FUNCTIONS
# ================================================================================================

def convert_to_json_serializable(obj):
    """
    Chuyển đổi numpy/pandas types sang Python native types để JSON serialization
    Xử lý các giá trị float đặc biệt (inf, -inf, nan) không hợp lệ trong JSON
    """
    import numpy as np
    import math

    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        # Convert numpy float to Python float
        val = float(obj)
        # Kiểm tra và xử lý các giá trị đặc biệt không hợp lệ trong JSON
        if math.isnan(val) or math.isinf(val):
            return None  # Hoặc có thể return 0, tùy vào yêu cầu
        return val
    elif isinstance(obj, float):
        # Xử lý Python float (không phải numpy)
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return convert_to_json_serializable(obj.tolist())
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def downsample_kaplan_meier(km_data: Dict[str, Any], max_points: int = 100) -> Dict[str, Any]:
    """
    Downsample Kaplan-Meier data để giảm kích thước response
    Giữ lại max_points điểm quan trọng nhất

    Args:
        km_data: Dict chứa timeline và survival_probabilities
        max_points: Số điểm tối đa muốn giữ lại

    Returns:
        Dict với downsampled data
    """
    if not km_data or 'timeline' not in km_data or 'survival_probabilities' not in km_data:
        return km_data

    timeline = km_data['timeline']
    survival_probs = km_data['survival_probabilities']

    # Nếu số điểm ít hơn max_points, không cần downsample
    if len(timeline) <= max_points:
        return km_data

    # Downsample: lấy mỗi n điểm
    step = len(timeline) // max_points
    if step < 1:
        step = 1

    # Luôn giữ điểm đầu và điểm cuối
    indices = list(range(0, len(timeline), step))
    if len(timeline) - 1 not in indices:
        indices.append(len(timeline) - 1)

    downsampled_timeline = [timeline[i] for i in indices]
    downsampled_probs = [survival_probs[i] for i in indices]

    print(f"🔽 [DOWNSAMPLE] Kaplan-Meier: {len(timeline)} điểm → {len(downsampled_timeline)} điểm")

    return {
        'timeline': downsampled_timeline,
        'survival_probabilities': downsampled_probs,
        'median_survival_time': km_data.get('median_survival_time'),
        'event_count': km_data.get('event_count'),
        'censored_count': km_data.get('censored_count'),
        'original_points': len(timeline),  # Thông tin về số điểm gốc
        'downsampled': True
    }


# ================================================================================================
# PYDANTIC MODELS
# ================================================================================================

class PredictionInput(BaseModel):
    """Model cho input dự báo (14 chỉ số X1-X14)"""
    X_1: float
    X_2: float
    X_3: float
    X_4: float
    X_5: float
    X_6: float
    X_7: float
    X_8: float
    X_9: float
    X_10: float
    X_11: float
    X_12: float
    X_13: float
    X_14: float


class GeminiAPIKeyRequest(BaseModel):
    """Model cho request set Gemini API key"""
    api_key: str


# ================================================================================================
# ENDPOINTS
# ================================================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Credit Risk Assessment API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    """
    Endpoint huấn luyện mô hình từ file CSV

    Args:
        file: File CSV chứa dữ liệu huấn luyện (phải có cột X_1 đến X_14 và cột 'default')

    Returns:
        Dict chứa thông tin huấn luyện và metrics
    """
    try:
        # Kiểm tra file extension
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File phải có định dạng CSV")

        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Huấn luyện mô hình
        result = credit_model.train(tmp_file_path)

        # Lưu mô hình
        credit_model.save_model("model_stacking.pkl")

        # Xóa file tạm
        os.unlink(tmp_file_path)

        return convert_to_json_serializable(result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi huấn luyện mô hình: {str(e)}")


@app.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Endpoint dự báo PD từ 14 chỉ số tài chính

    Args:
        input_data: Dict chứa 14 chỉ số X_1 đến X_14

    Returns:
        Dict chứa PD từ 4 models và kết quả dự đoán
    """
    try:
        # Kiểm tra mô hình đã được train chưa
        if credit_model.model is None:
            # Thử load model từ file
            if os.path.exists("model_stacking.pkl"):
                credit_model.load_model("model_stacking.pkl")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Mô hình chưa được huấn luyện. Vui lòng upload file CSV để huấn luyện trước."
                )

        # Chuyển input thành DataFrame
        input_dict = input_data.dict()
        X_new = pd.DataFrame([input_dict])

        # Dự báo
        result = credit_model.predict(X_new)

        return convert_to_json_serializable(result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự báo: {str(e)}")


@app.post("/predict-from-xlsx")
async def predict_from_xlsx(file: UploadFile = File(...)):
    """
    Endpoint dự báo PD từ file XLSX (3 sheets: CDKT, BCTN, LCTT)
    Tự động tính 14 chỉ số và chạy mô hình dự báo

    Args:
        file: File XLSX chứa 3 sheets (CDKT, BCTN, LCTT)

    Returns:
        Dict chứa 14 chỉ số và kết quả dự báo PD
    """
    try:
        # Kiểm tra file extension
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File phải có định dạng XLSX hoặc XLS")

        # Kiểm tra mô hình đã được train chưa
        if credit_model.model is None:
            if os.path.exists("model_stacking.pkl"):
                credit_model.load_model("model_stacking.pkl")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Mô hình chưa được huấn luyện. Vui lòng upload file CSV để huấn luyện trước."
                )

        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Đọc file XLSX
            excel_processor.read_excel(tmp_file_path)

            # Tính 14 chỉ số
            indicators = excel_processor.calculate_14_indicators()
            indicators_with_names = excel_processor.get_indicators_with_names()

            # Chuyển thành DataFrame để dự báo
            X_new = pd.DataFrame([indicators])

            # Dự báo PD
            prediction_result = credit_model.predict(X_new)

            # Trả về kết quả
            response_data = {
                "status": "success",
                "indicators": indicators_with_names,
                "indicators_dict": indicators,
                "prediction": prediction_result
            }

            return convert_to_json_serializable(response_data)
        finally:
            # Xóa file tạm trong finally block để đảm bảo file luôn được xóa
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass  # Bỏ qua lỗi khi xóa file

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file XLSX: {str(e)}")


@app.post("/analyze")
async def analyze_with_gemini(request_data: Dict[str, Any]):
    """
    Endpoint phân tích kết quả dự báo bằng Gemini API

    Args:
        request_data: Dict chứa kết quả dự báo và 14 chỉ số

    Returns:
        Dict chứa kết quả phân tích từ Gemini và khuyến nghị
    """
    try:
        # Lấy Gemini analyzer
        analyzer = get_gemini_analyzer()

        # Phân tích
        analysis = analyzer.analyze_credit_risk(request_data)

        return {
            "status": "success",
            "analysis": analysis
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy GEMINI_API_KEY. Vui lòng set biến môi trường. Chi tiết: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích bằng Gemini: {str(e)}")


@app.post("/analyze-industry")
async def analyze_industry(request_data: Dict[str, Any]):
    """
    Endpoint phân tích ngành nghề bằng Gemini API

    Args:
        request_data: Dict chứa industry code và industry_name

    Returns:
        Dict chứa kết quả phân tích ngành và dữ liệu charts
    """
    try:
        industry = request_data.get('industry', '')
        industry_name = request_data.get('industry_name', '')

        if not industry or not industry_name:
            raise HTTPException(
                status_code=400,
                detail="Thiếu thông tin industry hoặc industry_name"
            )

        # Lấy Gemini analyzer
        analyzer = get_gemini_analyzer()

        # Phân tích ngành
        result = analyzer.analyze_industry(industry, industry_name)

        return {
            "status": "success",
            "analysis": result["analysis"],
            "charts": result.get("charts", [])
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy GEMINI_API_KEY. Vui lòng set biến môi trường. Chi tiết: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích ngành: {str(e)}")


@app.post("/set-gemini-key")
async def set_gemini_key(request: GeminiAPIKeyRequest):
    """
    Endpoint để set Gemini API key

    Args:
        request: Dict chứa api_key

    Returns:
        Dict xác nhận
    """
    try:
        os.environ["GEMINI_API_KEY"] = request.api_key

        # Khởi tạo lại Gemini analyzer - cập nhật global instance
        from gemini_api import GeminiAnalyzer
        import gemini_api
        gemini_api.gemini_analyzer = GeminiAnalyzer(request.api_key)

        return {
            "status": "success",
            "message": "Gemini API key đã được set thành công"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi set Gemini API key: {str(e)}")


@app.post("/export-report")
async def export_report(report_data: Dict[str, Any]):
    """
    Endpoint xuất báo cáo Word

    Args:
        report_data: Dict chứa prediction, indicators, và analysis

    Returns:
        File Word báo cáo
    """
    try:
        # Tạo báo cáo
        report_gen = ReportGenerator()
        output_path = f"bao_cao_tin_dung_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"

        report_path = report_gen.generate_report(report_data, output_path)

        # Trả về file
        return FileResponse(
            path=report_path,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            filename=output_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xuất báo cáo: {str(e)}")


@app.post("/fetch-industry-data")
async def fetch_industry_data(request_data: Dict[str, Any]):
    """
    Endpoint để AI lấy dữ liệu ngành nghề tự động

    Args:
        request_data: Dict chứa industry code và industry_name

    Returns:
        Dict chứa dữ liệu ngành nghề
    """
    try:
        industry = request_data.get('industry', '')
        industry_name = request_data.get('industry_name', '')

        if not industry or not industry_name:
            raise HTTPException(
                status_code=400,
                detail="Thiếu thông tin industry hoặc industry_name"
            )

        # Lấy Gemini analyzer
        analyzer = get_gemini_analyzer()

        # Lấy dữ liệu
        result = analyzer.fetch_industry_data(industry, industry_name)

        return {
            "status": "success",
            "data": result.get("data", {})
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy GEMINI_API_KEY. Vui lòng set biến môi trường. Chi tiết: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu ngành: {str(e)}")


@app.post("/generate-charts")
async def generate_charts(request_data: Dict[str, Any]):
    """
    Endpoint tạo biểu đồ ECharts và phân tích sơ bộ

    Args:
        request_data: Dict chứa industry, industry_name, và data

    Returns:
        Dict chứa charts_data và brief_analysis
    """
    try:
        industry = request_data.get('industry', '')
        industry_name = request_data.get('industry_name', '')
        data = request_data.get('data', {})

        if not industry or not industry_name or not data:
            raise HTTPException(
                status_code=400,
                detail="Thiếu thông tin industry, industry_name hoặc data"
            )

        # Lấy Gemini analyzer
        analyzer = get_gemini_analyzer()

        # Tạo biểu đồ và phân tích
        result = analyzer.generate_charts_data(industry, industry_name, data)

        return {
            "status": "success",
            "charts_data": result.get("charts_data", []),
            "brief_analysis": result.get("brief_analysis", "")
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy GEMINI_API_KEY. Vui lòng set biến môi trường. Chi tiết: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo biểu đồ: {str(e)}")


@app.post("/deep-analyze-industry")
async def deep_analyze_industry_endpoint(request_data: Dict[str, Any]):
    """
    Endpoint phân tích sâu ảnh hưởng của ngành đến quyết định cho vay

    Args:
        request_data: Dict chứa industry, industry_name, data, và brief_analysis

    Returns:
        Dict chứa deep_analysis
    """
    try:
        industry = request_data.get('industry', '')
        industry_name = request_data.get('industry_name', '')
        data = request_data.get('data', {})
        brief_analysis = request_data.get('brief_analysis', '')

        if not industry or not industry_name or not data:
            raise HTTPException(
                status_code=400,
                detail="Thiếu thông tin industry, industry_name hoặc data"
            )

        # Lấy Gemini analyzer
        analyzer = get_gemini_analyzer()

        # Phân tích sâu
        deep_analysis = analyzer.deep_analyze_industry(industry, industry_name, data, brief_analysis)

        return {
            "status": "success",
            "deep_analysis": deep_analysis
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy GEMINI_API_KEY. Vui lòng set biến môi trường. Chi tiết: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích sâu: {str(e)}")


@app.post("/analyze-pd-with-industry")
async def analyze_pd_with_industry(request_data: Dict[str, Any]):
    """
    Endpoint phân tích PD kết hợp với ngành nghề

    Args:
        request_data: Dict chứa indicators_dict, industry, và industry_name

    Returns:
        Dict chứa phân tích chuyên sâu và charts_data
    """
    try:
        indicators_dict = request_data.get('indicators_dict', {})
        industry = request_data.get('industry', '')
        industry_name = request_data.get('industry_name', '')

        if not indicators_dict or not industry or not industry_name:
            raise HTTPException(
                status_code=400,
                detail="Thiếu thông tin indicators_dict, industry hoặc industry_name"
            )

        # Lấy Gemini analyzer
        analyzer = get_gemini_analyzer()

        # Phân tích PD kết hợp
        analysis = analyzer.analyze_pd_with_industry(indicators_dict, industry, industry_name)

        # Tạo biểu đồ từ 14 chỉ số
        charts_data = []

        # Biểu đồ 1: Radar chart cho 4 nhóm chỉ số chính
        charts_data.append({
            "title": {"text": "Tổng quan 14 Chỉ số Tài chính", "left": "center"},
            "tooltip": {},
            "radar": {
                "indicator": [
                    {"name": "Sinh lời (X1-X4)", "max": 1},
                    {"name": "Đòn bẩy (X5-X6)", "max": 5},
                    {"name": "Thanh toán (X7-X8)", "max": 5},
                    {"name": "Hiệu quả (X9-X14)", "max": 10}
                ]
            },
            "series": [{
                "type": "radar",
                "data": [{
                    "value": [
                        (indicators_dict.get('X_1', 0) + indicators_dict.get('X_2', 0) +
                         indicators_dict.get('X_3', 0) + indicators_dict.get('X_4', 0)) / 4,
                        (indicators_dict.get('X_5', 0) + indicators_dict.get('X_6', 0)) / 2,
                        (indicators_dict.get('X_7', 0) + indicators_dict.get('X_8', 0)) / 2,
                        (indicators_dict.get('X_9', 0) + indicators_dict.get('X_10', 0) +
                         indicators_dict.get('X_11', 0) + indicators_dict.get('X_12', 0) +
                         indicators_dict.get('X_14', 0)) / 5
                    ],
                    "name": "Chỉ số doanh nghiệp",
                    "areaStyle": {"color": "rgba(255, 107, 157, 0.3)"}
                }]
            }]
        })

        # Biểu đồ 2: Bar chart so sánh chỉ số sinh lời
        charts_data.append({
            "title": {"text": "Chỉ số Sinh lời (X1-X4)", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": ["Biên LN gộp (X1)", "Biên LN trước thuế (X2)", "ROA (X3)", "ROE (X4)"]
            },
            "yAxis": {"type": "value"},
            "series": [{
                "data": [
                    indicators_dict.get('X_1', 0),
                    indicators_dict.get('X_2', 0),
                    indicators_dict.get('X_3', 0),
                    indicators_dict.get('X_4', 0)
                ],
                "type": "bar",
                "itemStyle": {"color": "#10B981"},
                "label": {"show": True, "position": "top", "formatter": "{c}"}
            }]
        })

        # Biểu đồ 3: Bar chart chỉ số thanh toán & đòn bẩy
        charts_data.append({
            "title": {"text": "Thanh toán & Đòn bẩy (X5-X8)", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": ["Nợ/TS (X5)", "Nợ/VCSH (X6)", "TT hiện hành (X7)", "TT nhanh (X8)"]
            },
            "yAxis": {"type": "value"},
            "series": [{
                "data": [
                    indicators_dict.get('X_5', 0),
                    indicators_dict.get('X_6', 0),
                    indicators_dict.get('X_7', 0),
                    indicators_dict.get('X_8', 0)
                ],
                "type": "bar",
                "itemStyle": {"color": "#3B82F6"},
                "label": {"show": True, "position": "top", "formatter": "{c}"}
            }]
        })

        # Biểu đồ 4: Bar chart hiệu quả hoạt động
        charts_data.append({
            "title": {"text": "Hiệu quả Hoạt động (X9-X14)", "left": "center"},
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": ["Trả lãi (X9)", "Trả nợ gốc (X10)", "Tạo tiền (X11)",
                         "Vòng quay HTK (X12)", "Kỳ thu tiền (X13)", "Hiệu suất TS (X14)"]
            },
            "yAxis": {"type": "value"},
            "series": [{
                "data": [
                    indicators_dict.get('X_9', 0),
                    indicators_dict.get('X_10', 0),
                    indicators_dict.get('X_11', 0),
                    indicators_dict.get('X_12', 0),
                    indicators_dict.get('X_13', 0),
                    indicators_dict.get('X_14', 0)
                ],
                "type": "bar",
                "itemStyle": {"color": "#9C27B0"},
                "label": {"show": True, "position": "top", "formatter": "{c}"}
            }]
        })

        return {
            "status": "success",
            "analysis": analysis,
            "charts_data": charts_data
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy GEMINI_API_KEY. Vui lòng set biến môi trường. Chi tiết: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích PD kết hợp: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """
    Endpoint lấy thông tin mô hình hiện tại

    Returns:
        Dict chứa thông tin mô hình
    """
    try:
        if credit_model.model is None:
            # Thử load model từ file
            if os.path.exists("model_stacking.pkl"):
                credit_model.load_model("model_stacking.pkl")
            else:
                return {
                    "status": "not_trained",
                    "message": "Mô hình chưa được huấn luyện"
                }

        return {
            "status": "trained",
            "message": "Mô hình đã sẵn sàng",
            "metrics_train": credit_model.metrics_in,
            "metrics_test": credit_model.metrics_out
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thông tin mô hình: {str(e)}")


@app.post("/chat-assistant")
async def chat_assistant(data: Dict[str, Any]):
    """
    Endpoint chatbot - Trợ lý ảo trả lời câu hỏi về phân tích

    Args:
        data: Dict chứa question, context, indicators, prediction

    Returns:
        Dict chứa answer từ Gemini
    """
    try:
        question = data.get('question', '')
        context = data.get('context', '')
        indicators = data.get('indicators', {})
        prediction = data.get('prediction', {})

        if not question:
            raise HTTPException(status_code=400, detail="Thiếu câu hỏi (question)")

        # Lấy Gemini analyzer
        analyzer = get_gemini_analyzer()

        # Tạo prompt cho chatbot
        prompt = f"""
Bạn là Trợ lý ảo chuyên nghiệp của Agribank, chuyên trả lời các câu hỏi về phân tích rủi ro tín dụng.

**BỐI CẢNH PHÂN TÍCH TRƯỚC ĐÓ:**
{context}

**14 CHỈ SỐ TÀI CHÍNH:**
{str(indicators)}

**KẾT QUẢ DỰ BÁO PD:**
{str(prediction)}

**CÂU HỎI CỦA NGƯỜI DÙNG:**
{question}

**YÊU CẦU TRẢ LỜI:**
- Trả lời ngắn gọn, chính xác, dễ hiểu (100-200 từ)
- Dựa trên bối cảnh phân tích và dữ liệu đã có
- Nếu câu hỏi liên quan đến chỉ số tài chính, giải thích rõ ràng
- Nếu câu hỏi về khuyến nghị, đưa ra lời khuyên cụ thể
- Sử dụng tiếng Việt chuyên nghiệp

Hãy trả lời câu hỏi:
"""

        # Gọi Gemini API
        response = analyzer.model.generate_content(prompt)
        answer = response.text

        return {
            "status": "success",
            "answer": answer
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy GEMINI_API_KEY. Chi tiết: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")


@app.post("/simulate-scenario")
async def simulate_scenario(
    file: Optional[UploadFile] = File(None),
    indicators_json: Optional[str] = Form(None),
    scenario_type: str = Form("mild"),
    custom_revenue: float = Form(0),
    custom_interest: float = Form(0),
    custom_cogs: float = Form(0),
    custom_liquidity: float = Form(0)
):
    """
    Endpoint mô phỏng kịch bản xấu - Stress Testing với tính toán dây chuyền hoàn chỉnh (Phương án A)

    Args:
        file: File XLSX (nếu tải file mới) - Optional
        indicators_json: JSON string chứa 14 chỉ số (nếu dùng dữ liệu từ Tab Dự báo PD) - Optional
        scenario_type: Loại kịch bản ("mild", "moderate", "crisis", "custom")
        custom_revenue: % thay đổi doanh thu thuần (chỉ dùng khi scenario_type="custom")
        custom_interest: % thay đổi lãi suất vay (chỉ dùng khi scenario_type="custom")
        custom_cogs: % thay đổi giá vốn hàng bán (chỉ dùng khi scenario_type="custom")
        custom_liquidity: % sốc thanh khoản TSNH (chỉ dùng khi scenario_type="custom")

    Returns:
        Dict chứa:
        - indicators_before: 14 chỉ số trước khi áp kịch bản
        - indicators_after: 14 chỉ số sau khi áp kịch bản
        - prediction_before: PD trước khi áp kịch bản
        - prediction_after: PD sau khi áp kịch bản
        - pd_change_pct: % thay đổi PD
        - scenario_info: Thông tin về kịch bản đã áp dụng
    """
    try:
        import json

        # Kiểm tra mô hình đã được train chưa
        if credit_model.model is None:
            if os.path.exists("model_stacking.pkl"):
                credit_model.load_model("model_stacking.pkl")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Mô hình chưa được huấn luyện. Vui lòng upload file CSV để huấn luyện trước."
                )

        # 1. LẤY 14 CHỈ SỐ BAN ĐẦU (indicators_before)
        indicators_before = {}

        if file:
            # Trường hợp 1: Tải file XLSX mới
            if not file.filename.endswith(('.xlsx', '.xls')):
                raise HTTPException(status_code=400, detail="File phải có định dạng XLSX hoặc XLS")

            # Lưu file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Đọc file XLSX và tính 14 chỉ số
                excel_processor.read_excel(tmp_file_path)
                indicators_before = excel_processor.calculate_14_indicators()
            finally:
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass

        elif indicators_json:
            # Trường hợp 2: Sử dụng dữ liệu từ Tab Dự báo PD
            indicators_before = json.loads(indicators_json)
        else:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp file XLSX hoặc dữ liệu từ Tab Dự báo PD"
            )

        # 2. XÁC ĐỊNH % BIẾN ĐỘNG THEO KỊCH BẢN (PHƯƠNG ÁN A - STRESS TESTING)
        scenario_configs = {
            "mild": {
                "name": "🟠 Kinh tế giảm nhẹ",
                "revenue_change": -5,
                "interest_rate_change": 10,
                "cogs_change": 3,
                "liquidity_shock": -5
            },
            "moderate": {
                "name": "🔴 Cú sốc kinh tế trung bình",
                "revenue_change": -12,
                "interest_rate_change": 25,
                "cogs_change": 8,
                "liquidity_shock": -12
            },
            "crisis": {
                "name": "⚫ Khủng hoảng",
                "revenue_change": -25,
                "interest_rate_change": 40,
                "cogs_change": 15,
                "liquidity_shock": -25
            },
            "custom": {
                "name": "🟡 Tùy chọn biến động",
                "revenue_change": custom_revenue,
                "interest_rate_change": custom_interest,
                "cogs_change": custom_cogs,
                "liquidity_shock": custom_liquidity
            }
        }

        if scenario_type not in scenario_configs:
            raise HTTPException(
                status_code=400,
                detail=f"Loại kịch bản không hợp lệ. Chọn: {', '.join(scenario_configs.keys())}"
            )

        scenario = scenario_configs[scenario_type]

        # 3. TÍNH 14 CHỈ SỐ SAU KHI ÁP KỊCH BẢN (indicators_after)
        # Sử dụng PHƯƠNG ÁN A: Stress Testing với tính toán dây chuyền hoàn chỉnh
        indicators_after = excel_processor.simulate_scenario_full_propagation(
            original_indicators=indicators_before,
            revenue_change_pct=scenario["revenue_change"],
            interest_rate_change_pct=scenario["interest_rate_change"],
            cogs_change_pct=scenario["cogs_change"],
            liquidity_shock_pct=scenario["liquidity_shock"]
        )

        # 4. DỰ BÁO PD TRƯỚC VÀ SAU
        # Dự báo PD trước khi áp kịch bản
        X_before = pd.DataFrame([indicators_before])
        prediction_before = credit_model.predict(X_before)

        # Dự báo PD sau khi áp kịch bản
        X_after = pd.DataFrame([indicators_after])
        prediction_after = credit_model.predict(X_after)

        # 5. TÍNH % THAY ĐỔI PD
        pd_before = prediction_before["pd_stacking"]
        pd_after = prediction_after["pd_stacking"]
        pd_change_pct = ((pd_after - pd_before) / pd_before * 100) if pd_before != 0 else 0

        # 6. CHUẨN BỊ KẾT QUẢ TRẢ VỀ
        # Chuyển đổi indicators thành list có tên
        def indicators_to_list(indicators_dict):
            indicator_names = {
                'X_1': 'Hệ số biên lợi nhuận gộp',
                'X_2': 'Hệ số biên lợi nhuận trước thuế',
                'X_3': 'Tỷ suất lợi nhuận trước thuế trên tổng tài sản (ROA)',
                'X_4': 'Tỷ suất lợi nhuận trước thuế trên vốn chủ sở hữu (ROE)',
                'X_5': 'Hệ số nợ trên tài sản',
                'X_6': 'Hệ số nợ trên vốn chủ sở hữu',
                'X_7': 'Khả năng thanh toán hiện hành',
                'X_8': 'Khả năng thanh toán nhanh',
                'X_9': 'Hệ số khả năng trả lãi',
                'X_10': 'Hệ số khả năng trả nợ gốc',
                'X_11': 'Hệ số khả năng tạo tiền trên vốn chủ sở hữu',
                'X_12': 'Vòng quay hàng tồn kho',
                'X_13': 'Kỳ thu tiền bình quân',
                'X_14': 'Hiệu suất sử dụng tài sản'
            }
            result = []
            for key, value in indicators_dict.items():
                result.append({
                    'code': key,
                    'name': indicator_names[key],
                    'value': value
                })
            return result

        return {
            "status": "success",
            "scenario_info": {
                "type": scenario_type,
                "name": scenario["name"],
                "changes": {
                    "revenue": scenario["revenue_change"],
                    "interest": scenario["interest_rate_change"],
                    "cogs": scenario["cogs_change"],
                    "liquidity": scenario["liquidity_shock"]
                }
            },
            "indicators_before": indicators_to_list(indicators_before),
            "indicators_before_dict": indicators_before,
            "indicators_after": indicators_to_list(indicators_after),
            "indicators_after_dict": indicators_after,
            "prediction_before": prediction_before,
            "prediction_after": prediction_after,
            "pd_change": {
                "before": pd_before,
                "after": pd_after,
                "change_pct": round(pd_change_pct, 2),
                "change_absolute": round(pd_after - pd_before, 6)
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi mô phỏng kịch bản: {str(e)}")


@app.post("/analyze-scenario")
async def analyze_scenario(request_data: Dict[str, Any]):
    """
    Endpoint phân tích kết quả mô phỏng kịch bản bằng Gemini API

    Args:
        request_data: Dict chứa kết quả mô phỏng kịch bản

    Returns:
        Dict chứa kết quả phân tích từ Gemini
    """
    try:
        # Lấy Gemini analyzer
        analyzer = get_gemini_analyzer()

        # Phân tích kịch bản
        analysis = analyzer.analyze_scenario_simulation(request_data)

        return {
            "status": "success",
            "analysis": analysis
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy GEMINI_API_KEY. Vui lòng set biến môi trường. Chi tiết: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích kịch bản bằng Gemini: {str(e)}")


@app.post("/simulate-scenario-macro")
async def simulate_scenario_macro(
    file: Optional[UploadFile] = File(None),
    indicators_json: Optional[str] = Form(None),
    scenario_type: str = Form("recession_mild"),
    industry_code: str = Form("manufacturing"),
    custom_gdp: float = Form(0),
    custom_cpi: float = Form(0),
    custom_ppi: float = Form(0),
    custom_policy_rate: float = Form(0),
    custom_fx: float = Form(0)
):
    """
    Endpoint mô phỏng kịch bản vĩ mô (Macro Stress Testing)

    Args:
        file: File XLSX (nếu tải file mới) - Optional
        indicators_json: JSON string chứa 14 chỉ số (nếu dùng dữ liệu từ Tab Dự báo PD) - Optional
        scenario_type: Loại kịch bản ("recession_mild", "recession_moderate", "crisis", "custom")
        industry_code: Mã ngành ("manufacturing", "export", "retail")
        custom_gdp: % tăng trưởng GDP (chỉ dùng khi scenario_type="custom")
        custom_cpi: % lạm phát CPI (chỉ dùng khi scenario_type="custom")
        custom_ppi: % lạm phát PPI (chỉ dùng khi scenario_type="custom")
        custom_policy_rate: Thay đổi lãi suất NHNN bps (chỉ dùng khi scenario_type="custom")
        custom_fx: % thay đổi tỷ giá USD/VND (chỉ dùng khi scenario_type="custom")

    Returns:
        Dict chứa:
        - macro_variables: 5 biến vĩ mô đã chọn
        - micro_shocks: 4 biến vi mô được tính từ kênh truyền dẫn
        - indicators_before: 14 chỉ số trước khi áp kịch bản
        - indicators_after: 14 chỉ số sau khi áp kịch bản
        - prediction_before: PD trước khi áp kịch bản
        - prediction_after: PD sau khi áp kịch bản
        - pd_change_pct: % thay đổi PD
        - scenario_info: Thông tin về kịch bản đã áp dụng
    """
    try:
        import json

        # Kiểm tra mô hình đã được train chưa
        if credit_model.model is None:
            if os.path.exists("model_stacking.pkl"):
                credit_model.load_model("model_stacking.pkl")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Mô hình chưa được huấn luyện. Vui lòng upload file CSV để huấn luyện trước."
                )

        # 1. LẤY 14 CHỈ SỐ BAN ĐẦU (indicators_before)
        indicators_before = {}

        if file:
            # Trường hợp 1: Tải file XLSX mới
            if not file.filename.endswith(('.xlsx', '.xls')):
                raise HTTPException(status_code=400, detail="File phải có định dạng XLSX hoặc XLS")

            # Lưu file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Đọc file XLSX và tính 14 chỉ số
                excel_processor.read_excel(tmp_file_path)
                indicators_before = excel_processor.calculate_14_indicators()
            finally:
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass

        elif indicators_json:
            # Trường hợp 2: Sử dụng dữ liệu từ Tab Dự báo PD
            indicators_before = json.loads(indicators_json)
        else:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp file XLSX hoặc dữ liệu từ Tab Dự báo PD"
            )

        # 2. XÁC ĐỊNH 5 BIẾN VĨ MÔ THEO KỊCH BẢN
        macro_scenario_configs = {
            "recession_mild": {
                "name": "🟠 Suy thoái nhẹ",
                "gdp_growth_pct": -1.5,
                "inflation_cpi_pct": 6.0,
                "inflation_ppi_pct": 8.0,
                "policy_rate_change_bps": 100,
                "fx_usd_vnd_pct": 3.0
            },
            "recession_moderate": {
                "name": "🔴 Suy thoái trung bình",
                "gdp_growth_pct": -3.5,
                "inflation_cpi_pct": 10.0,
                "inflation_ppi_pct": 14.0,
                "policy_rate_change_bps": 200,
                "fx_usd_vnd_pct": 6.0
            },
            "crisis": {
                "name": "⚫ Khủng hoảng",
                "gdp_growth_pct": -6.0,
                "inflation_cpi_pct": 15.0,
                "inflation_ppi_pct": 20.0,
                "policy_rate_change_bps": 300,
                "fx_usd_vnd_pct": 10.0
            },
            "custom": {
                "name": "🟡 Tùy chỉnh vĩ mô",
                "gdp_growth_pct": custom_gdp,
                "inflation_cpi_pct": custom_cpi,
                "inflation_ppi_pct": custom_ppi,
                "policy_rate_change_bps": custom_policy_rate,
                "fx_usd_vnd_pct": custom_fx
            }
        }

        if scenario_type not in macro_scenario_configs:
            raise HTTPException(
                status_code=400,
                detail=f"Loại kịch bản không hợp lệ. Chọn: {', '.join(macro_scenario_configs.keys())}"
            )

        macro_scenario = macro_scenario_configs[scenario_type]

        # 3. KÊNH TRUYỀN DẪN: MACRO → MICRO
        # Gọi function macro_to_micro_transmission()
        micro_shocks = excel_processor.macro_to_micro_transmission(
            gdp_growth_pct=macro_scenario["gdp_growth_pct"],
            inflation_cpi_pct=macro_scenario["inflation_cpi_pct"],
            inflation_ppi_pct=macro_scenario["inflation_ppi_pct"],
            policy_rate_change_bps=macro_scenario["policy_rate_change_bps"],
            fx_usd_vnd_pct=macro_scenario["fx_usd_vnd_pct"],
            industry_code=industry_code
        )

        # 4. TÍNH 14 CHỈ SỐ SAU KHI ÁP 4 BIẾN VI MÔ
        # Sử dụng simulate_scenario_full_propagation() với 4 biến vi mô
        indicators_after = excel_processor.simulate_scenario_full_propagation(
            original_indicators=indicators_before,
            revenue_change_pct=micro_shocks["revenue_change_pct"],
            interest_rate_change_pct=micro_shocks["interest_rate_change_pct"],
            cogs_change_pct=micro_shocks["cogs_change_pct"],
            liquidity_shock_pct=micro_shocks["liquidity_shock_pct"]
        )

        # 5. DỰ BÁO PD TRƯỚC VÀ SAU
        # Dự báo PD trước khi áp kịch bản
        X_before = pd.DataFrame([indicators_before])
        prediction_before = credit_model.predict(X_before)

        # Dự báo PD sau khi áp kịch bản
        X_after = pd.DataFrame([indicators_after])
        prediction_after = credit_model.predict(X_after)

        # 6. TÍNH % THAY ĐỔI PD
        pd_before = prediction_before["pd_stacking"]
        pd_after = prediction_after["pd_stacking"]
        pd_change_pct = ((pd_after - pd_before) / pd_before * 100) if pd_before != 0 else 0

        # 7. CHUẨN BỊ KẾT QUẢ TRẢ VỀ
        # Chuyển đổi indicators thành list có tên
        def indicators_to_list(indicators_dict):
            indicator_names = {
                'X_1': 'Hệ số biên lợi nhuận gộp',
                'X_2': 'Hệ số biên lợi nhuận trước thuế',
                'X_3': 'Tỷ suất lợi nhuận trước thuế trên tổng tài sản (ROA)',
                'X_4': 'Tỷ suất lợi nhuận trước thuế trên vốn chủ sở hữu (ROE)',
                'X_5': 'Hệ số nợ trên tài sản',
                'X_6': 'Hệ số nợ trên vốn chủ sở hữu',
                'X_7': 'Khả năng thanh toán hiện hành',
                'X_8': 'Khả năng thanh toán nhanh',
                'X_9': 'Hệ số khả năng trả lãi',
                'X_10': 'Hệ số khả năng trả nợ gốc',
                'X_11': 'Hệ số khả năng tạo tiền trên vốn chủ sở hữu',
                'X_12': 'Vòng quay hàng tồn kho',
                'X_13': 'Kỳ thu tiền bình quân',
                'X_14': 'Hiệu suất sử dụng tài sản'
            }
            result = []
            for key, value in indicators_dict.items():
                result.append({
                    'code': key,
                    'name': indicator_names[key],
                    'value': value
                })
            return result

        # Tên ngành nghề
        industry_names = {
            "manufacturing": "Sản xuất",
            "export": "Xuất khẩu",
            "retail": "Bán lẻ"
        }

        return {
            "status": "success",
            "scenario_info": {
                "type": scenario_type,
                "name": macro_scenario["name"],
                "industry": industry_names.get(industry_code, industry_code)
            },
            "macro_variables": {
                "gdp_growth_pct": macro_scenario["gdp_growth_pct"],
                "inflation_cpi_pct": macro_scenario["inflation_cpi_pct"],
                "inflation_ppi_pct": macro_scenario["inflation_ppi_pct"],
                "policy_rate_change_bps": macro_scenario["policy_rate_change_bps"],
                "fx_usd_vnd_pct": macro_scenario["fx_usd_vnd_pct"]
            },
            "micro_shocks": micro_shocks,
            "indicators_before": indicators_to_list(indicators_before),
            "indicators_before_dict": indicators_before,
            "indicators_after": indicators_to_list(indicators_after),
            "indicators_after_dict": indicators_after,
            "prediction_before": prediction_before,
            "prediction_after": prediction_after,
            "pd_change": {
                "before": pd_before,
                "after": pd_after,
                "change_pct": round(pd_change_pct, 2),
                "change_absolute": round(pd_after - pd_before, 6)
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi mô phỏng kịch bản vĩ mô: {str(e)}")


@app.post("/analyze-macro")
async def analyze_macro(request_data: Dict[str, Any]):
    """
    Endpoint phân tích kết quả mô phỏng vĩ mô bằng Gemini API

    Args:
        request_data: Dict chứa kết quả mô phỏng vĩ mô

    Returns:
        Dict chứa kết quả phân tích từ Gemini
    """
    try:
        # Lấy Gemini analyzer
        analyzer = get_gemini_analyzer()

        # Lấy thông tin từ request
        scenario_info = request_data.get('scenario_info', {})
        macro_variables = request_data.get('macro_variables', {})
        micro_shocks = request_data.get('micro_shocks', {})
        indicators_before = request_data.get('indicators_before_dict', {})
        indicators_after = request_data.get('indicators_after_dict', {})
        pd_change = request_data.get('pd_change', {})

        # Tạo prompt cho Gemini
        prompt = f"""
Bạn là chuyên gia phân tích kinh tế vĩ mô và rủi ro tín dụng của Agribank. Hãy phân tích kết quả mô phỏng kịch bản vĩ mô dưới đây.

**THÔNG TIN KỊCH BẢN VĨ MÔ:**

**Kịch bản:** {scenario_info.get('name', 'N/A')}
**Ngành:** {scenario_info.get('industry', 'N/A')}

**5 BIẾN VĨ MÔ:**
- Tăng trưởng GDP: {macro_variables.get('gdp_growth_pct', 0):.1f}%
- Lạm phát CPI: {macro_variables.get('inflation_cpi_pct', 0):.1f}%
- Lạm phát PPI: {macro_variables.get('inflation_ppi_pct', 0):.1f}%
- Thay đổi lãi suất NHNN: {macro_variables.get('policy_rate_change_bps', 0):.0f} bps
- Thay đổi tỷ giá USD/VND: {macro_variables.get('fx_usd_vnd_pct', 0):.1f}%

**4 BIẾN VI MÔ (Kênh truyền dẫn):**
- Thay đổi doanh thu: {micro_shocks.get('revenue_change_pct', 0):.2f}%
- Thay đổi lãi suất vay: {micro_shocks.get('interest_rate_change_pct', 0):.2f}%
- Thay đổi giá vốn hàng bán: {micro_shocks.get('cogs_change_pct', 0):.2f}%
- Sốc thanh khoản: {micro_shocks.get('liquidity_shock_pct', 0):.2f}%

**TÁC ĐỘNG ĐẾN XÁC SUẤT VỠ NỢ:**
- PD trước: {pd_change.get('before', 0):.4f}
- PD sau: {pd_change.get('after', 0):.4f}
- Thay đổi: {pd_change.get('change_pct', 0):.2f}% (tuyệt đối: {pd_change.get('change_absolute', 0):.4f})

**YÊU CẦU PHÂN TÍCH:**

Hãy viết báo cáo phân tích chi tiết (sử dụng Markdown) với cấu trúc sau:

## 📊 TỔNG QUAN KỊCH BẢN VĨ MÔ
(2-3 câu mô tả kịch bản vĩ mô và mức độ nghiêm trọng)

## 🔄 PHÂN TÍCH KÊNH TRUYỀN DẪN
(Giải thích cách 5 biến vĩ mô tác động lên 4 biến vi mô của doanh nghiệp)

### Tác động lên Doanh thu
(Phân tích chi tiết)

### Tác động lên Chi phí & Lãi suất
(Phân tích chi tiết)

### Tác động lên Thanh khoản
(Phân tích chi tiết)

## 📈 ĐÁNH GIÁ TÁC ĐỘNG ĐẾN PD

### Mức độ thay đổi
(Phân tích mức độ thay đổi PD: nhẹ/trung bình/nghiêm trọng)

### Các chỉ số tài chính chịu ảnh hưởng nhiều nhất
(Liệt kê 3-5 chỉ số bị ảnh hưởng mạnh nhất)

## 💡 KHUYẾN NGHỊ

### Đối với Doanh nghiệp
(2-3 khuyến nghị cụ thể)

### Đối với Ngân hàng
(2-3 khuyến nghị về chính sách tín dụng)

## ⚠️ RỦI RO CẦN LƯU Ý
(Liệt kê 2-3 rủi ro tiềm ẩn cần theo dõi)

---
**Lưu ý:** Viết ngắn gọn, chuyên nghiệp, dễ hiểu. Tập trung vào insights và actionable recommendations.
"""

        # Gọi Gemini API
        response = analyzer.model.generate_content(prompt)
        analysis = response.text

        return {
            "status": "success",
            "analysis": analysis
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy GEMINI_API_KEY. Vui lòng set biến môi trường. Chi tiết: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích vĩ mô bằng Gemini: {str(e)}")


@app.post("/train-early-warning")
async def train_early_warning_model(file: UploadFile = File(...)):
    """
    Endpoint huấn luyện Early Warning System

    Args:
        file: File Excel chứa 1300 DN với 14 chỉ số (X_1 → X_14) + cột 'label' (0=không vỡ nợ, 1=vỡ nợ)

    Returns:
        Dict chứa thông tin về training:
        - status: success
        - num_samples: Số lượng mẫu
        - feature_importances: Feature importances từ RandomForest
        - cluster_distribution: Phân bố các cluster
    """
    try:
        # Kiểm tra file extension
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(
                status_code=400,
                detail="File phải có định dạng XLSX, XLS hoặc CSV"
            )

        # Lưu file tạm
        suffix = '.xlsx' if file.filename.endswith(('.xlsx', '.xls')) else '.csv'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Đọc file
            if suffix == '.csv':
                df = pd.read_csv(tmp_file_path)
            else:
                df = pd.read_excel(tmp_file_path)

            # Kiểm tra các cột cần thiết
            required_cols = [f'X_{i}' for i in range(1, 15)] + ['label']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"File thiếu các cột: {', '.join(missing_cols)}"
                )

            # Train Early Warning System
            result = early_warning_system.train_models(df)

            response_data = {
                "status": "success",
                "message": "Early Warning System trained successfully!",
                **result
            }

            return convert_to_json_serializable(response_data)

        finally:
            # Xóa file tạm
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi train Early Warning System: {str(e)}")


@app.post("/early-warning-check")
async def early_warning_check(
    file: Optional[UploadFile] = File(None),
    indicators_json: Optional[str] = Form(None),
    report_period: Optional[str] = Form(None),
    industry_code: str = Form("manufacturing")
):
    """
    Endpoint kiểm tra cảnh báo rủi ro sớm

    Args:
        file: File Excel (nếu tải file mới) - Optional
        indicators_json: JSON string chứa 14 chỉ số (nếu dùng dữ liệu từ Tab Dự báo PD) - Optional
        report_period: Kỳ báo cáo (Quý/6 tháng/Năm) - Optional, chỉ để hiển thị
        industry_code: Mã ngành ("manufacturing", "export", "retail")

    Returns:
        Dict chứa:
        - health_score: Health Score (0-100)
        - risk_level: Mức rủi ro (Safe/Watch/Warning/Alert)
        - risk_level_color: Màu sắc
        - current_pd: PD hiện tại
        - top_weaknesses: Top 3 điểm yếu
        - cluster_info: Thông tin cluster
        - pd_projection: Dự báo PD tương lai
        - gemini_diagnosis: Báo cáo chẩn đoán từ Gemini AI
        - feature_importances: Feature importances
    """
    try:
        import json

        # Kiểm tra Early Warning System đã được train chưa
        if early_warning_system.stacking_model is None:
            raise HTTPException(
                status_code=400,
                detail="Early Warning System chưa được train. Vui lòng upload file training data trước."
            )

        # Kiểm tra mô hình PD đã được train chưa
        if credit_model.model is None:
            if os.path.exists("model_stacking.pkl"):
                credit_model.load_model("model_stacking.pkl")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Mô hình PD chưa được huấn luyện. Vui lòng train mô hình trước."
                )

        # 1. LẤY 14 CHỈ SỐ
        indicators = {}

        if file:
            # Trường hợp 1: Tải file XLSX mới
            if not file.filename.endswith(('.xlsx', '.xls')):
                raise HTTPException(status_code=400, detail="File phải có định dạng XLSX hoặc XLS")

            # Lưu file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Đọc file XLSX và tính 14 chỉ số
                excel_processor.read_excel(tmp_file_path)
                indicators = excel_processor.calculate_14_indicators()
            finally:
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass

        elif indicators_json:
            # Trường hợp 2: Sử dụng dữ liệu từ Tab Dự báo PD
            indicators = json.loads(indicators_json)
        else:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp file XLSX hoặc dữ liệu từ Tab Dự báo PD"
            )

        # 2. TÍNH HEALTH SCORE
        health_score = early_warning_system.calculate_health_score(indicators)

        # 3. PHÂN LOẠI MỨC RỦI RO
        risk_info = early_warning_system.classify_risk_level(health_score)

        # 4. TÍNH PD HIỆN TẠI (sử dụng early_warning_system.stacking_model)
        feature_cols = [f'X_{i}' for i in range(1, 15)]
        X_current = [[indicators[col] for col in feature_cols]]
        current_pd = early_warning_system.stacking_model.predict_proba(X_current)[0, 1] * 100

        # 5. PHÁT HIỆN ĐIỂM YẾU
        weaknesses = early_warning_system.detect_weaknesses(indicators)

        # 6. XÁC ĐỊNH VỊ TRÍ CLUSTER
        cluster_info = early_warning_system.get_cluster_position(indicators)

        # 7. DỰ BÁO PD TƯƠNG LAI (3/6/12 tháng x 3 kịch bản)
        scenarios = ['recession_mild', 'recession_moderate', 'crisis']
        time_periods = [3, 6, 12]

        pd_projection = {
            'current': current_pd
        }

        for scenario in scenarios:
            pd_projection[scenario] = {}
            for months in time_periods:
                pd_future = early_warning_system.project_future_pd(
                    indicators=indicators,
                    months=months,
                    scenario=scenario,
                    excel_processor=excel_processor,
                    industry_code=industry_code
                )
                pd_projection[scenario][f'{months}_months'] = pd_future

        # 8. TẠO BÁO CÁO CHẨN ĐOÁN BẰNG GEMINI AI
        gemini_diagnosis = early_warning_system.generate_gemini_diagnosis(
            health_score=health_score,
            risk_info=risk_info,
            weaknesses=weaknesses,
            cluster_info=cluster_info,
            pd_projections=pd_projection,
            current_pd=current_pd,
            gemini_api_key=GEMINI_API_KEY
        )

        # 9. TRẢ VỀ KẾT QUẢ
        response_data = {
            "status": "success",
            "health_score": health_score,
            "risk_level": risk_info['risk_level'],
            "risk_level_color": risk_info['risk_level_color'],
            "risk_level_icon": risk_info['risk_level_icon'],
            "risk_level_text": risk_info['risk_level_text'],
            "current_pd": current_pd,
            "top_weaknesses": weaknesses,
            "cluster_info": cluster_info,
            "pd_projection": pd_projection,
            "gemini_diagnosis": gemini_diagnosis,
            "feature_importances": early_warning_system.feature_importances,
            "report_period": report_period,
            "indicators": indicators  # Thêm 14 chỉ số để frontend có thể vẽ biểu đồ radar
        }

        return convert_to_json_serializable(response_data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi kiểm tra cảnh báo rủi ro: {str(e)}")


@app.post("/train-anomaly")
async def train_anomaly_model(file: UploadFile = File(...)):
    """
    Endpoint huấn luyện Anomaly Detection System

    Args:
        file: File Excel/CSV chứa 1300 DN với 14 chỉ số (X_1 → X_14) + cột 'label' (0=khỏe mạnh, 1=vỡ nợ)

    Returns:
        Dict chứa thông tin về training:
        - status: success
        - feature_statistics: Thống kê 14 features (P5, P25, P50, P75, P95)
        - contamination_rate: Tỷ lệ contamination
    """
    try:
        # Kiểm tra file extension
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(
                status_code=400,
                detail="File phải có định dạng XLSX, XLS hoặc CSV"
            )

        # Lưu file tạm
        suffix = '.xlsx' if file.filename.endswith(('.xlsx', '.xls')) else '.csv'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Đọc file
            if suffix == '.csv':
                df = pd.read_csv(tmp_file_path)
            else:
                df = pd.read_excel(tmp_file_path)

            # Kiểm tra các cột cần thiết
            required_cols = [f'X_{i}' for i in range(1, 15)] + ['label']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"File thiếu các cột: {', '.join(missing_cols)}"
                )

            # Train Anomaly Detection System
            result = anomaly_system.train_model(df)

            response_data = {
                "status": "success",
                "message": "Anomaly Detection System trained successfully!",
                **result
            }

            return convert_to_json_serializable(response_data)

        finally:
            # Xóa file tạm
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi train Anomaly Detection System: {str(e)}")


@app.post("/check-anomaly")
async def check_anomaly(
    file: Optional[UploadFile] = File(None),
    indicators_json: Optional[str] = Form(None)
):
    """
    Endpoint kiểm tra bất thường cho DN mới

    Args:
        file: File Excel (nếu tải file mới) - Optional
        indicators_json: JSON string chứa 14 chỉ số (nếu dùng dữ liệu từ Tab Dự báo PD) - Optional

    Returns:
        Dict chứa:
        - anomaly_score: Điểm bất thường (0-100)
        - risk_level: Mức rủi ro
        - abnormal_features: List các features bất thường
        - anomaly_type: Loại bất thường
        - gemini_explanation: Giải thích từ Gemini AI
        - comparison_with_healthy: So sánh với DN khỏe mạnh
    """
    try:
        import json

        # Kiểm tra Anomaly Detection System đã được train chưa
        if anomaly_system.model is None:
            raise HTTPException(
                status_code=400,
                detail="Anomaly Detection System chưa được train. Vui lòng upload file training data trước."
            )

        # 1. LẤY 14 CHỈ SỐ
        indicators = {}

        if file:
            # Trường hợp 1: Tải file XLSX mới
            if not file.filename.endswith(('.xlsx', '.xls')):
                raise HTTPException(status_code=400, detail="File phải có định dạng XLSX hoặc XLS")

            # Lưu file tạm
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Đọc file XLSX và tính 14 chỉ số
                excel_processor.read_excel(tmp_file_path)
                indicators = excel_processor.calculate_14_indicators()
            finally:
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass

        elif indicators_json:
            # Trường hợp 2: Sử dụng dữ liệu từ Tab Dự báo PD
            indicators = json.loads(indicators_json)
        else:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp file XLSX hoặc dữ liệu từ Tab Dự báo PD"
            )

        # 2. TÍNH ANOMALY SCORE
        anomaly_score = anomaly_system.calculate_anomaly_score(indicators)

        # 3. PHÁT HIỆN CÁC FEATURES BẤT THƯỜNG
        abnormal_features = anomaly_system.detect_abnormal_features(indicators)

        # 4. PHÂN LOẠI LOẠI BẤT THƯỜNG
        anomaly_type = anomaly_system.classify_anomaly_type(indicators, abnormal_features)

        # 5. XÁC ĐỊNH MỨC RỦI RO
        if anomaly_score < 60:
            risk_level = "Bình thường"
            risk_level_color = "#10B981"
            risk_level_icon = "⚠️"
        elif anomaly_score < 80:
            risk_level = "Bất thường Trung bình"
            risk_level_color = "#F59E0B"
            risk_level_icon = "🔶"
        else:
            risk_level = "Bất thường Cao"
            risk_level_color = "#EF4444"
            risk_level_icon = "🔴"

        # 6. TẠO GIẢI THÍCH BẰNG GEMINI AI
        gemini_explanation = anomaly_system.generate_gemini_explanation(
            indicators=indicators,
            anomaly_score=anomaly_score,
            abnormal_features=abnormal_features,
            anomaly_type=anomaly_type,
            gemini_api_key=GEMINI_API_KEY
        )

        # 7. SO SÁNH VỚI DN KHỎE MẠNH (cho Radar Chart)
        comparison_with_healthy = []
        for feature in anomaly_system.feature_names:
            comparison_with_healthy.append({
                'feature': anomaly_system.indicator_names[feature],
                'current': indicators[feature],
                'healthy_mean': anomaly_system.healthy_stats[feature]['mean']
            })

        # 8. TRẢ VỀ KẾT QUẢ
        response_data = {
            "status": "success",
            "anomaly_score": anomaly_score,
            "risk_level": risk_level,
            "risk_level_color": risk_level_color,
            "risk_level_icon": risk_level_icon,
            "abnormal_features": abnormal_features,
            "anomaly_type": anomaly_type,
            "gemini_explanation": gemini_explanation,
            "comparison_with_healthy": comparison_with_healthy,
            "indicators": indicators
        }

        return convert_to_json_serializable(response_data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi kiểm tra bất thường: {str(e)}")


# ================================================================================================
# SURVIVAL ANALYSIS ENDPOINTS
# ================================================================================================

@app.post("/train-survival")
async def train_survival_models(file: UploadFile = File(...)):
    """
    Huấn luyện Survival Analysis Models (Cox PH + Random Survival Forest)
    Chạy tuần tự 2 mô hình để đảm bảo ổn định

    Input: CSV/Excel file với cột:
    - X_1 đến X_14: 14 chỉ số tài chính
    - months_to_default: Thời gian đến khi vỡ nợ (tháng)
    - event: 1 = vỡ nợ, 0 = censored (chưa vỡ nợ)

    Returns:
    - Training metrics (C-index, log-likelihood)
    - Kaplan-Meier baseline survival function (downsampled để giảm kích thước)
    - Hazard ratios
    """
    print("\n" + "="*80)
    print("🚀 [SURVIVAL TRAINING] Bắt đầu huấn luyện Cox PH & RSF models...")
    print("="*80)

    tmp_file_path = None

    try:
        # 1. LƯU FILE TẠM THỜI
        print("📁 [SURVIVAL TRAINING] Đang lưu file tạm thời...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        print(f"✅ [SURVIVAL TRAINING] Đã lưu file: {tmp_file_path}")

        try:
            # 2. ĐỌC DỮ LIỆU
            print("📊 [SURVIVAL TRAINING] Đang đọc dữ liệu...")
            if file.filename.endswith('.csv'):
                df = pd.read_csv(tmp_file_path)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(tmp_file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="File phải là định dạng CSV hoặc Excel (.xlsx, .xls)"
                )

            print(f"✅ [SURVIVAL TRAINING] Đã đọc {len(df)} dòng dữ liệu")

            # 3. KIỂM TRA CỘT CẦN THIẾT
            print("🔍 [SURVIVAL TRAINING] Đang kiểm tra các cột dữ liệu...")
            required_features = [f'X_{i}' for i in range(1, 15)]
            required_cols = required_features + ['months_to_default']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Thiếu các cột: {', '.join(missing_cols)}"
                )

            # Nếu không có cột 'event', tự động tạo (giả định tất cả đều vỡ nợ)
            if 'event' not in df.columns:
                df['event'] = 1
                print("⚠️  [SURVIVAL TRAINING] Không tìm thấy cột 'event', tạo tự động (all events = 1)")

            print(f"✅ [SURVIVAL TRAINING] Validation hoàn tất. Events: {int(df['event'].sum())}, Censored: {int((1-df['event']).sum())}")

            # 4. HUẤN LUYỆN TUẦN TỰ (SEQUENTIAL) - ĐƠN GIẢN VÀ ỔN ĐỊNH HƠN
            print("\n⚡ [SURVIVAL TRAINING] Đang huấn luyện lần lượt Cox PH và RSF models...")
            cox_result = None
            rsf_result = None
            training_errors = []

            # 4.1. TRAIN COX PH MODEL TRƯỚC
            try:
                print("🔄 [COX MODEL - 1/2] Bắt đầu huấn luyện Cox Proportional Hazards Model...")
                cox_result = survival_system.train_cox_model(
                    df,
                    duration_col='months_to_default',
                    event_col='event'
                )
                print(f"✅ [COX MODEL - 1/2] Hoàn thành! C-index: {cox_result['c_index']:.4f}")
            except Exception as e:
                print(f"❌ [COX MODEL - 1/2] Lỗi: {str(e)}")
                training_errors.append({
                    "model": "Cox PH",
                    "error": str(e)
                })

            # 4.2. TRAIN RSF MODEL SAU
            try:
                print("🔄 [RSF MODEL - 2/2] Bắt đầu huấn luyện Random Survival Forest Model...")
                rsf_result = survival_system.train_random_survival_forest(
                    df,
                    duration_col='months_to_default',
                    event_col='event',
                    n_estimators=100
                )
                print(f"✅ [RSF MODEL - 2/2] Hoàn thành! C-index: {rsf_result['c_index']:.4f}")
            except Exception as e:
                print(f"❌ [RSF MODEL - 2/2] Lỗi: {str(e)}")
                training_errors.append({
                    "model": "RSF",
                    "error": str(e)
                })

            print("\n" + "="*80)
            print("📊 [SURVIVAL TRAINING] Kết quả huấn luyện tuần tự:")
            print(f"   - Cox PH: {'✅ Thành công' if cox_result else '❌ Thất bại'}")
            print(f"   - RSF: {'✅ Thành công' if rsf_result else '❌ Thất bại'}")
            print("="*80)

            # 6. TÍNH KAPLAN-MEIER BASELINE (chỉ khi có ít nhất 1 model thành công)
            km_result = None
            if cox_result or rsf_result:
                try:
                    print("📈 [SURVIVAL TRAINING] Đang tính Kaplan-Meier baseline...")
                    km_result = survival_system.calculate_kaplan_meier(
                        df,
                        duration_col='months_to_default',
                        event_col='event'
                    )

                    # Downsample KM data để giảm kích thước response
                    if km_result and 'timeline' in km_result:
                        original_points = len(km_result.get('timeline', []))
                        km_result = downsample_kaplan_meier(km_result, max_points=100)
                        print(f"✅ [SURVIVAL TRAINING] Kaplan-Meier baseline đã tính xong ({original_points} → {len(km_result.get('timeline', []))} điểm)")
                    else:
                        print("✅ [SURVIVAL TRAINING] Kaplan-Meier baseline đã tính xong")
                except Exception as e:
                    print(f"⚠️  [SURVIVAL TRAINING] Không thể tính Kaplan-Meier: {str(e)}")
                    km_result = {"error": str(e)}

            # 7. LẤY HAZARD RATIOS (chỉ khi Cox model thành công)
            hazard_ratios = None
            if cox_result:
                try:
                    print("📊 [SURVIVAL TRAINING] Đang tính hazard ratios...")
                    hazard_ratios = survival_system.get_hazard_ratios(top_k=14)
                    print(f"✅ [SURVIVAL TRAINING] Đã tính {len(hazard_ratios)} hazard ratios")
                except Exception as e:
                    print(f"⚠️  [SURVIVAL TRAINING] Không thể tính hazard ratios: {str(e)}")
                    hazard_ratios = []

            # 8. LƯU MODELS (chỉ khi có ít nhất 1 model thành công)
            if cox_result or rsf_result:
                try:
                    print("💾 [SURVIVAL TRAINING] Đang lưu models...")
                    survival_system.save_models('survival_models.pkl')
                    print("✅ [SURVIVAL TRAINING] Models đã được lưu vào survival_models.pkl")
                except Exception as e:
                    print(f"⚠️  [SURVIVAL TRAINING] Không thể lưu models: {str(e)}")

            # 9. TRẢ VỀ KẾT QUẢ (JSON SERIALIZABLE)
            print("\n" + "="*80)
            print("🎉 [SURVIVAL TRAINING] Hoàn thành quá trình training!")
            print("="*80 + "\n")

            # Kiểm tra xem có ít nhất 1 model thành công không
            if not cox_result and not rsf_result:
                raise HTTPException(
                    status_code=500,
                    detail=f"Cả 2 models đều thất bại. Errors: {training_errors}"
                )

            # Tạo response data và convert sang JSON serializable types
            response_data = {
                "status": "success",
                "message": "Đã huấn luyện thành công các mô hình Survival Analysis",
                "cox_model": cox_result if cox_result else {"status": "failed", "error": "Training failed"},
                "rsf_model": rsf_result if rsf_result else {"status": "failed", "error": "Training failed"},
                "kaplan_meier": km_result if km_result else {"status": "not_computed"},
                "hazard_ratios": hazard_ratios if hazard_ratios else [],
                "training_errors": training_errors,
                "n_samples": len(df),
                "n_events": int(df['event'].sum()),
                "n_censored": int((1 - df['event']).sum())
            }

            # Convert tất cả numpy/pandas types sang Python native types
            print("🔄 [SURVIVAL TRAINING] Đang serialize response data...")
            json_response = convert_to_json_serializable(response_data)

            # Log kích thước response (ước tính)
            import json
            try:
                response_size = len(json.dumps(json_response))
                print(f"📦 [SURVIVAL TRAINING] Response size: {response_size:,} bytes ({response_size/1024:.2f} KB)")
            except Exception as e:
                print(f"⚠️  [SURVIVAL TRAINING] Không thể tính kích thước response: {str(e)}")

            print("✅ [SURVIVAL TRAINING] Response đã sẵn sàng để gửi về frontend\n")

            # Sử dụng JSONResponse explicitly để đảm bảo serialize chính xác
            return JSONResponse(content=json_response)

        finally:
            # Xóa file tạm
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    print(f"🗑️  [SURVIVAL TRAINING] Đã xóa file tạm: {tmp_file_path}")
                except Exception as e:
                    print(f"⚠️  [SURVIVAL TRAINING] Không thể xóa file tạm: {str(e)}")

    except HTTPException as e:
        print(f"❌ [SURVIVAL TRAINING] HTTPException: {str(e)}")
        raise
    except Exception as e:
        print(f"❌ [SURVIVAL TRAINING] Lỗi không mong muốn: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi huấn luyện survival models: {str(e)}"
        )


@app.post("/train-cox")
async def train_cox_model(file: UploadFile = File(...)):
    """
    Huấn luyện riêng Cox Proportional Hazards Model
    (Endpoint đơn giản hơn, chỉ train 1 model để tránh lỗi network)

    Input: CSV/Excel file với cột:
    - X_1 đến X_14: 14 chỉ số tài chính
    - months_to_default: Thời gian đến khi vỡ nợ (tháng)
    - event: 1 = vỡ nợ, 0 = censored (chưa vỡ nợ)

    Returns:
    - Cox Model training metrics (C-index, log-likelihood)
    - Kaplan-Meier baseline survival function
    - Hazard ratios
    """
    print("\n" + "="*80)
    print("🚀 [COX TRAINING] Bắt đầu huấn luyện Cox PH model...")
    print("="*80)

    tmp_file_path = None

    try:
        # 1. LƯU FILE TẠM THỜI
        print("📁 [COX TRAINING] Đang lưu file tạm thời...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        print(f"✅ [COX TRAINING] Đã lưu file: {tmp_file_path}")

        try:
            # 2. ĐỌC DỮ LIỆU
            print("📊 [COX TRAINING] Đang đọc dữ liệu...")
            if file.filename.endswith('.csv'):
                df = pd.read_csv(tmp_file_path)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(tmp_file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="File phải là định dạng CSV hoặc Excel (.xlsx, .xls)"
                )

            print(f"✅ [COX TRAINING] Đã đọc {len(df)} dòng dữ liệu")

            # 3. KIỂM TRA CỘT CẦN THIẾT
            print("🔍 [COX TRAINING] Đang kiểm tra các cột dữ liệu...")
            required_features = [f'X_{i}' for i in range(1, 15)]
            required_cols = required_features + ['months_to_default']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Thiếu các cột: {', '.join(missing_cols)}"
                )

            # Nếu không có cột 'event', tự động tạo
            if 'event' not in df.columns:
                df['event'] = 1
                print("⚠️  [COX TRAINING] Không tìm thấy cột 'event', tạo tự động (all events = 1)")

            print(f"✅ [COX TRAINING] Validation hoàn tất. Events: {int(df['event'].sum())}, Censored: {int((1-df['event']).sum())}")

            # 4. HUẤN LUYỆN COX MODEL
            print("🔄 [COX TRAINING] Đang huấn luyện Cox Proportional Hazards Model...")
            cox_result = survival_system.train_cox_model(
                df,
                duration_col='months_to_default',
                event_col='event'
            )
            print(f"✅ [COX TRAINING] Hoàn thành! C-index: {cox_result['c_index']:.4f}")

            # 5. TÍNH KAPLAN-MEIER BASELINE
            km_result = None
            try:
                print("📈 [COX TRAINING] Đang tính Kaplan-Meier baseline...")
                km_result = survival_system.calculate_kaplan_meier(
                    df,
                    duration_col='months_to_default',
                    event_col='event'
                )

                # Downsample KM data
                if km_result and 'timeline' in km_result:
                    original_points = len(km_result.get('timeline', []))
                    km_result = downsample_kaplan_meier(km_result, max_points=100)
                    print(f"✅ [COX TRAINING] Kaplan-Meier baseline đã tính xong ({original_points} → {len(km_result.get('timeline', []))} điểm)")
            except Exception as e:
                print(f"⚠️  [COX TRAINING] Không thể tính Kaplan-Meier: {str(e)}")
                km_result = {"error": str(e)}

            # 6. LẤY HAZARD RATIOS
            hazard_ratios = None
            try:
                print("📊 [COX TRAINING] Đang tính hazard ratios...")
                hazard_ratios = survival_system.get_hazard_ratios(top_k=14)
                print(f"✅ [COX TRAINING] Đã tính {len(hazard_ratios)} hazard ratios")
            except Exception as e:
                print(f"⚠️  [COX TRAINING] Không thể tính hazard ratios: {str(e)}")
                hazard_ratios = []

            # 7. LƯU MODEL
            try:
                print("💾 [COX TRAINING] Đang lưu model...")
                survival_system.save_models('survival_models.pkl')
                print("✅ [COX TRAINING] Model đã được lưu vào survival_models.pkl")
            except Exception as e:
                print(f"⚠️  [COX TRAINING] Không thể lưu model: {str(e)}")

            # 8. TẠO RESPONSE
            print("\n" + "="*80)
            print("🎉 [COX TRAINING] Hoàn thành quá trình training!")
            print("="*80 + "\n")

            response_data = {
                "status": "success",
                "message": "Đã huấn luyện thành công Cox Proportional Hazards Model",
                "cox_model": cox_result,
                "kaplan_meier": km_result if km_result else {"status": "not_computed"},
                "hazard_ratios": hazard_ratios if hazard_ratios else [],
                "n_samples": len(df),
                "n_events": int(df['event'].sum()),
                "n_censored": int((1 - df['event']).sum())
            }

            # Convert sang JSON serializable
            print("🔄 [COX TRAINING] Đang serialize response data...")
            json_response = convert_to_json_serializable(response_data)

            # Log kích thước
            import json
            try:
                response_size = len(json.dumps(json_response))
                print(f"📦 [COX TRAINING] Response size: {response_size:,} bytes ({response_size/1024:.2f} KB)")
            except Exception as e:
                print(f"⚠️  [COX TRAINING] Không thể tính kích thước response: {str(e)}")

            print("✅ [COX TRAINING] Response đã sẵn sàng để gửi về frontend\n")

            return JSONResponse(content=json_response)

        finally:
            # Xóa file tạm
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    print(f"🗑️  [COX TRAINING] Đã xóa file tạm: {tmp_file_path}")
                except Exception as e:
                    print(f"⚠️  [COX TRAINING] Không thể xóa file tạm: {str(e)}")

    except HTTPException as e:
        print(f"❌ [COX TRAINING] HTTPException: {str(e)}")
        raise
    except Exception as e:
        print(f"❌ [COX TRAINING] Lỗi không mong muốn: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi huấn luyện Cox model: {str(e)}"
        )


@app.post("/train-rsf")
async def train_rsf_model(file: UploadFile = File(...)):
    """
    Huấn luyện riêng Random Survival Forest Model
    (Endpoint đơn giản hơn, chỉ train 1 model để tránh lỗi network)

    Input: CSV/Excel file với cột:
    - X_1 đến X_14: 14 chỉ số tài chính
    - months_to_default: Thời gian đến khi vỡ nợ (tháng)
    - event: 1 = vỡ nợ, 0 = censored (chưa vỡ nợ)

    Returns:
    - RSF Model training metrics (C-index)
    """
    print("\n" + "="*80)
    print("🚀 [RSF TRAINING] Bắt đầu huấn luyện Random Survival Forest model...")
    print("="*80)

    tmp_file_path = None

    try:
        # 1. LƯU FILE TẠM THỜI
        print("📁 [RSF TRAINING] Đang lưu file tạm thời...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        print(f"✅ [RSF TRAINING] Đã lưu file: {tmp_file_path}")

        try:
            # 2. ĐỌC DỮ LIỆU
            print("📊 [RSF TRAINING] Đang đọc dữ liệu...")
            if file.filename.endswith('.csv'):
                df = pd.read_csv(tmp_file_path)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(tmp_file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="File phải là định dạng CSV hoặc Excel (.xlsx, .xls)"
                )

            print(f"✅ [RSF TRAINING] Đã đọc {len(df)} dòng dữ liệu")

            # 3. KIỂM TRA CỘT CẦN THIẾT
            print("🔍 [RSF TRAINING] Đang kiểm tra các cột dữ liệu...")
            required_features = [f'X_{i}' for i in range(1, 15)]
            required_cols = required_features + ['months_to_default']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Thiếu các cột: {', '.join(missing_cols)}"
                )

            # Nếu không có cột 'event', tự động tạo
            if 'event' not in df.columns:
                df['event'] = 1
                print("⚠️  [RSF TRAINING] Không tìm thấy cột 'event', tạo tự động (all events = 1)")

            print(f"✅ [RSF TRAINING] Validation hoàn tất. Events: {int(df['event'].sum())}, Censored: {int((1-df['event']).sum())}")

            # 4. HUẤN LUYỆN RSF MODEL
            print("🔄 [RSF TRAINING] Đang huấn luyện Random Survival Forest Model...")
            rsf_result = survival_system.train_random_survival_forest(
                df,
                duration_col='months_to_default',
                event_col='event',
                n_estimators=100
            )
            print(f"✅ [RSF TRAINING] Hoàn thành! C-index: {rsf_result['c_index']:.4f}")

            # 5. LƯU MODEL
            try:
                print("💾 [RSF TRAINING] Đang lưu model...")
                survival_system.save_models('survival_models.pkl')
                print("✅ [RSF TRAINING] Model đã được lưu vào survival_models.pkl")
            except Exception as e:
                print(f"⚠️  [RSF TRAINING] Không thể lưu model: {str(e)}")

            # 6. TẠO RESPONSE
            print("\n" + "="*80)
            print("🎉 [RSF TRAINING] Hoàn thành quá trình training!")
            print("="*80 + "\n")

            response_data = {
                "status": "success",
                "message": "Đã huấn luyện thành công Random Survival Forest Model",
                "rsf_model": rsf_result,
                "n_samples": len(df),
                "n_events": int(df['event'].sum()),
                "n_censored": int((1 - df['event']).sum())
            }

            # Convert sang JSON serializable
            print("🔄 [RSF TRAINING] Đang serialize response data...")
            json_response = convert_to_json_serializable(response_data)

            # Log kích thước
            import json
            try:
                response_size = len(json.dumps(json_response))
                print(f"📦 [RSF TRAINING] Response size: {response_size:,} bytes ({response_size/1024:.2f} KB)")
            except Exception as e:
                print(f"⚠️  [RSF TRAINING] Không thể tính kích thước response: {str(e)}")

            print("✅ [RSF TRAINING] Response đã sẵn sàng để gửi về frontend\n")

            return JSONResponse(content=json_response)

        finally:
            # Xóa file tạm
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    print(f"🗑️  [RSF TRAINING] Đã xóa file tạm: {tmp_file_path}")
                except Exception as e:
                    print(f"⚠️  [RSF TRAINING] Không thể xóa file tạm: {str(e)}")

    except HTTPException as e:
        print(f"❌ [RSF TRAINING] HTTPException: {str(e)}")
        raise
    except Exception as e:
        print(f"❌ [RSF TRAINING] Lỗi không mong muốn: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi huấn luyện RSF model: {str(e)}"
        )


@app.post("/predict-survival")
async def predict_survival(
    file: Optional[UploadFile] = File(None),
    indicators_json: Optional[str] = Form(None)
):
    """
    Dự báo Survival Curve cho một doanh nghiệp mới

    Input:
    - file: XLSX file (3 sheets: CDKT, BCTN, LCTT)
    - indicators_json: JSON string với 14 chỉ số

    Returns:
    - Survival curve (timeline + probabilities)
    - Median time-to-default
    - Survival probabilities tại 6/12/24 tháng
    - Risk classification
    - Hazard ratios (top 5 quan trọng nhất) - Model-level metrics
    - Risk contributions (top 5) - Individual-level metrics cho doanh nghiệp này
    """
    try:
        import json

        # 1. LẤY 14 CHỈ SỐ TÀI CHÍNH
        if file:
            # Từ file XLSX
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(await file.read())
                tmp_file_path = tmp_file.name

            try:
                excel_processor.read_excel(tmp_file_path)
                indicators = excel_processor.calculate_14_indicators()
            finally:
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass

        elif indicators_json:
            # Từ JSON
            indicators = json.loads(indicators_json)
        else:
            raise HTTPException(
                status_code=400,
                detail="Vui lòng cung cấp file XLSX hoặc dữ liệu 14 chỉ số"
            )

        # 2. KIỂM TRA MODEL ĐÃ ĐƯỢC HUẤN LUYỆN
        if survival_system.cox_model is None:
            raise HTTPException(
                status_code=400,
                detail="Mô hình chưa được huấn luyện. Vui lòng gọi /train-survival trước."
            )

        # 3. DỰ BÁO SURVIVAL CURVE (sử dụng Cox model)
        survival_curve = survival_system.predict_survival_curve(
            indicators=indicators,
            model_type='cox'
        )

        # 4. TÍNH MEDIAN TIME-TO-DEFAULT
        median_time = survival_system.calculate_median_time_to_default(
            indicators=indicators,
            model_type='cox'
        )

        # 5. TÍNH SURVIVAL PROBABILITIES TẠI CÁC THỜI ĐIỂM CỤ THỂ
        survival_probs = survival_system.get_survival_probabilities_at_times(
            indicators=indicators,
            times=[6, 12, 24],
            model_type='cox'
        )

        # 6. PHÂN LOẠI RỦI RO
        risk_info = survival_system.get_risk_classification(median_time)

        # 7. LẤY HAZARD RATIOS (TOP 5) - Model-level metrics
        hazard_ratios = survival_system.get_hazard_ratios(top_k=5)

        # 8. LẤY INDIVIDUAL RISK CONTRIBUTIONS (TOP 5) - CỤ THỂ CHO DOANH NGHIỆP NÀY
        # KHÁC với hazard ratios (model-level, giống nhau cho mọi DN)
        risk_contributions = survival_system.get_individual_risk_contributions(
            indicators=indicators,
            top_k=5
        )

        # 9. TẠO CẢNH BÁO NẾU RỦI RO CAO
        warning = None
        if median_time < 12:
            warning = {
                'type': 'HIGH_RISK',
                'message': f'⚠️ CẢNH BÁO: Median time-to-default chỉ {median_time:.1f} tháng! Doanh nghiệp có nguy cơ vỡ nợ rất cao.',
                'recommendation': 'Ngân hàng nên từ chối cho vay hoặc yêu cầu tài sản thế chấp bổ sung.'
            }
        elif median_time < 18:
            warning = {
                'type': 'MEDIUM_RISK',
                'message': f'⚠️ LƯU Ý: Median time-to-default là {median_time:.1f} tháng. Cần theo dõi chặt chẽ.',
                'recommendation': 'Xem xét hạn mức tín dụng thấp hơn và yêu cầu báo cáo tài chính định kỳ.'
            }

        response_data = {
            "status": "success",
            "indicators": indicators,
            "survival_curve": survival_curve,
            "median_time_to_default": float(median_time),
            "survival_probabilities": survival_probs,
            "risk_classification": risk_info,
            "hazard_ratios": hazard_ratios,  # Model-level (giống cho mọi DN)
            "risk_contributions": risk_contributions,  # CỤ THỂ cho doanh nghiệp này
            "warning": warning
        }

        return convert_to_json_serializable(response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi dự báo survival: {str(e)}"
        )


@app.get("/survival-metrics")
async def get_survival_metrics():
    """
    Lấy các metrics và hazard ratios từ trained models

    Returns:
    - Cox model metrics (C-index, log-likelihood)
    - RSF model metrics
    - Hazard ratios cho tất cả 14 chỉ số
    - Kaplan-Meier baseline survival
    """
    try:
        # Kiểm tra model đã được huấn luyện
        if survival_system.cox_model is None:
            raise HTTPException(
                status_code=400,
                detail="Mô hình chưa được huấn luyện. Vui lòng gọi /train-survival trước."
            )

        # Lấy hazard ratios (tất cả 14 chỉ số)
        hazard_ratios = survival_system.get_hazard_ratios(top_k=14)

        # Lấy Kaplan-Meier baseline
        km_baseline = survival_system.calculate_kaplan_meier()

        response_data = {
            "status": "success",
            "metrics": survival_system.metrics,
            "hazard_ratios": hazard_ratios,
            "kaplan_meier_baseline": km_baseline
        }

        return convert_to_json_serializable(response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy survival metrics: {str(e)}"
        )


@app.post("/analyze-survival-gemini")
async def analyze_survival_gemini(
    data: Dict[str, Any]
):
    """
    Phân tích kết quả Survival Analysis bằng Gemini AI

    Input:
    - data: Dict chứa survival analysis results

    Returns:
    - Dict với analysis text từ Gemini
    """
    try:
        # Lấy data từ request
        survival_data = data.get('data', {})

        # Phân tích bằng Gemini
        analyzer = get_gemini_analyzer(GEMINI_API_KEY)
        analysis = analyzer.analyze_survival_results(survival_data)

        response_data = {
            "status": "success",
            "analysis": analysis
        }

        return convert_to_json_serializable(response_data)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích survival bằng Gemini: {str(e)}"
        )


@app.post("/export-survival-report")
async def export_survival_report(
    data: Dict[str, Any]
):
    """
    Xuất báo cáo Survival Analysis ra file Word

    Input:
    - data: Dict chứa survival analysis results và Gemini analysis

    Returns:
    - File Word báo cáo
    """
    try:
        # Tạo tên file unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Bao_cao_Survival_Analysis_{timestamp}.docx"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        # Tạo báo cáo
        report_path = report_generator.generate_survival_report(data, filepath)

        # Trả về file
        return FileResponse(
            path=report_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xuất báo cáo: {str(e)}"
        )


# ================================================================================================
# MAIN
# ================================================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
