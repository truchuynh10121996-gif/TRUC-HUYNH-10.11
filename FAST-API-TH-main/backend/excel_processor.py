"""
Module xử lý file XLSX và tính toán 14 chỉ số tài chính
File XLSX phải có 3 sheets: CDKT (Cân đối kế toán), BCTN (Báo cáo thu nhập), LCTT (Lưu chuyển tiền tệ)
"""

import pandas as pd
from typing import Dict, Any
import numpy as np
import re


class ExcelProcessor:
    """Class xử lý file XLSX và tính toán 14 chỉ số tài chính"""

    def __init__(self):
        self.cdkt_df = None  # Cân đối kế toán
        self.bctn_df = None  # Báo cáo thu nhập
        self.lctt_df = None  # Lưu chuyển tiền tệ
        self.financial_indicators = {}

    def read_excel(self, file_path: str) -> bool:
        """
        Đọc file XLSX với 3 sheets

        Args:
            file_path: Đường dẫn file XLSX

        Returns:
            True nếu đọc thành công, False nếu thất bại
        """
        try:
            # Đọc 3 sheets với context manager để đảm bảo file được đóng
            with pd.ExcelFile(file_path) as excel_file:
                # Kiểm tra các sheet cần thiết
                required_sheets = ['CDKT', 'BCTN', 'LCTT']
                available_sheets = excel_file.sheet_names

                missing_sheets = [sheet for sheet in required_sheets if sheet not in available_sheets]
                if missing_sheets:
                    raise ValueError(f"Thiếu các sheet: {', '.join(missing_sheets)}. File phải có 3 sheets: CDKT, BCTN, LCTT")

                # Đọc dữ liệu từng sheet
                self.cdkt_df = excel_file.parse('CDKT')
                self.bctn_df = excel_file.parse('BCTN')
                self.lctt_df = excel_file.parse('LCTT')

            return True

        except Exception as e:
            raise ValueError(f"Lỗi khi đọc file XLSX: {str(e)}")

    def get_value_from_sheet(self, df: pd.DataFrame, indicator_name: str, column_index: int = -1) -> float:
        """
        Lấy giá trị từ sheet dựa trên tên chỉ tiêu và cột
        Giả định: Cột đầu tiên là tên chỉ tiêu, cột CUỐI CÙNG là giá trị năm gần nhất (cuối kỳ)

        Args:
            df: DataFrame chứa dữ liệu
            indicator_name: Tên chỉ tiêu cần tìm
            column_index: Chỉ số cột cần lấy (-1 = cuối cùng, -2 = trước cuối cùng)

        Returns:
            Giá trị của chỉ tiêu
        """
        try:
            # Tìm trong cột đầu tiên (chỉ tiêu)
            col_name = df.columns[0]
            # Lấy cột theo chỉ số: -1 = cuối cùng (cuối kỳ), -2 = trước cuối cùng (đầu kỳ)
            if len(df.columns) > abs(column_index):
                value_col = df.columns[column_index]
            else:
                value_col = df.columns[-1]  # Fallback nếu không đủ cột

            # Chuẩn hóa indicator_name để tìm kiếm tốt hơn
            # Loại bỏ số thứ tự ở đầu (VD: "1. Tiền" -> "tiền")
            search_name = indicator_name.lower().strip()
            # Loại bỏ các ký tự số và dấu chấm ở đầu
            search_name = re.sub(r'^\d+\.\s*', '', search_name)

            # Tìm dòng có chứa indicator_name (case-insensitive, loại bỏ khoảng trắng)
            # Áp dụng cùng chuẩn hóa cho từng dòng trong DataFrame
            def normalize_text(text):
                text = str(text).strip().lower()
                # Loại bỏ số thứ tự ở đầu
                text = re.sub(r'^\d+\.\s*', '', text)
                return text

            mask = df[col_name].apply(normalize_text).str.contains(
                search_name, na=False, regex=False
            )

            if mask.any():
                value = df.loc[mask, value_col].iloc[0]
                # Xử lý giá trị
                if pd.isna(value):
                    return 0.0

                # Chuyển đổi giá trị sang float - xử lý nhiều định dạng
                try:
                    # Nếu đã là số thì return luôn
                    if isinstance(value, (int, float)):
                        return float(value)

                    # Chuyển sang string để xử lý
                    value_str = str(value).strip()

                    # Loại bỏ các ký tự không phải số (trừ dấu âm và dấu thập phân)
                    # Xử lý format: "1,000,000.50" hoặc "1.000.000,50" hoặc "(1000)" (số âm)

                    # Kiểm tra nếu có dấu ngoặc đơn (số âm)
                    is_negative = False
                    if value_str.startswith('(') and value_str.endswith(')'):
                        is_negative = True
                        value_str = value_str[1:-1]

                    # Kiểm tra nếu có dấu trừ
                    if value_str.startswith('-'):
                        is_negative = True
                        value_str = value_str[1:]

                    # Loại bỏ khoảng trắng, ký tự đặc biệt
                    value_str = value_str.replace(' ', '').replace('\xa0', '')

                    # Xác định dấu thập phân (dấu cuối cùng trong chuỗi)
                    # Nếu có cả dấu phẩy và dấu chấm, dấu nào xuất hiện sau là dấu thập phân
                    if ',' in value_str and '.' in value_str:
                        # Tìm vị trí cuối cùng của mỗi dấu
                        last_comma = value_str.rfind(',')
                        last_dot = value_str.rfind('.')

                        if last_comma > last_dot:
                            # Dấu phẩy là thập phân (định dạng châu Âu: 1.000,50)
                            value_str = value_str.replace('.', '').replace(',', '.')
                        else:
                            # Dấu chấm là thập phân (định dạng Mỹ: 1,000.50)
                            value_str = value_str.replace(',', '')
                    elif ',' in value_str:
                        # Chỉ có dấu phẩy
                        # Kiểm tra xem có phải phân cách hàng nghìn không
                        parts = value_str.split(',')
                        if len(parts) == 2 and len(parts[1]) <= 2:
                            # Có thể là thập phân (VD: 1000,50)
                            value_str = value_str.replace(',', '.')
                        else:
                            # Là phân cách hàng nghìn (VD: 1,000,000)
                            value_str = value_str.replace(',', '')

                    # Chuyển sang float
                    float_value = float(value_str)

                    # Áp dụng dấu âm nếu cần
                    if is_negative:
                        float_value = -float_value

                    print(f"✅ Tìm thấy '{indicator_name}': {float_value}")
                    return float_value

                except (ValueError, AttributeError) as e:
                    print(f"⚠️ Không thể chuyển đổi giá trị '{value}' cho '{indicator_name}': {str(e)}")
                    return 0.0
            else:
                print(f"⚠️ Không tìm thấy chỉ tiêu '{indicator_name}' trong sheet")
                return 0.0

        except Exception as e:
            print(f"❌ Lỗi khi lấy giá trị {indicator_name}: {str(e)}")
            return 0.0

    def get_average_from_two_periods(self, df: pd.DataFrame, indicator_name: str) -> float:
        """
        Lấy giá trị bình quân từ 2 kỳ: cuối kỳ (cột cuối) và đầu kỳ (cột trước cuối)

        Args:
            df: DataFrame chứa dữ liệu
            indicator_name: Tên chỉ tiêu cần tìm

        Returns:
            Giá trị bình quân của 2 kỳ
        """
        # Lấy giá trị cuối kỳ (cột cuối cùng)
        cuoi_ky = self.get_value_from_sheet(df, indicator_name, column_index=-1)

        # Lấy giá trị đầu kỳ (cột trước cuối cùng)
        dau_ky = self.get_value_from_sheet(df, indicator_name, column_index=-2)

        # Tính bình quân
        binh_quan = (cuoi_ky + dau_ky) / 2

        print(f"📊 {indicator_name}: Đầu kỳ={dau_ky:.2f}, Cuối kỳ={cuoi_ky:.2f}, Bình quân={binh_quan:.2f}")

        return binh_quan

    def calculate_14_indicators(self) -> Dict[str, float]:
        """
        Tính toán 14 chỉ số tài chính từ 3 sheets

        Returns:
            Dict chứa 14 chỉ số X_1 đến X_14
        """
        if self.cdkt_df is None or self.bctn_df is None or self.lctt_df is None:
            raise ValueError("Chưa đọc dữ liệu từ file XLSX. Vui lòng gọi read_excel() trước.")

        # Lấy các chỉ tiêu từ BCTN (Báo cáo thu nhập)
        doanh_thu_thuan = self.get_value_from_sheet(self.bctn_df, "doanh thu thuần")
        if doanh_thu_thuan == 0:
            doanh_thu_thuan = self.get_value_from_sheet(self.bctn_df, "doanh thu bán")

        loi_nhuan_gop = self.get_value_from_sheet(self.bctn_df, "lợi nhuận gộp")
        gia_von_hang_ban = self.get_value_from_sheet(self.bctn_df, "giá vốn")

        # ✅ THAY ĐỔI: Lấy "Lợi nhuận trước thuế" từ LCTT thay vì BCTN
        loi_nhuan_truoc_thue = self.get_value_from_sheet(self.lctt_df, "lợi nhuận trước thuế")

        # Lấy các chỉ tiêu từ CDKT (Cân đối kế toán)
        # ✅ THAY ĐỔI: Lấy giá trị bình quân tự động từ 2 cột cuối (đầu kỳ và cuối kỳ)
        tong_tai_san = self.get_value_from_sheet(self.cdkt_df, "tổng tài sản", column_index=-1)
        binh_quan_tong_tai_san = self.get_average_from_two_periods(self.cdkt_df, "tổng tài sản")

        von_chu_so_huu = self.get_value_from_sheet(self.cdkt_df, "vốn chủ sở hữu", column_index=-1)
        binh_quan_von_chu_so_huu = self.get_average_from_two_periods(self.cdkt_df, "vốn chủ sở hữu")

        no_phai_tra = self.get_value_from_sheet(self.cdkt_df, "nợ phải trả")
        if no_phai_tra == 0:
            no_phai_tra = self.get_value_from_sheet(self.cdkt_df, "tổng nợ")

        tai_san_ngan_han = self.get_value_from_sheet(self.cdkt_df, "tài sản ngắn hạn", column_index=-1)
        no_ngan_han = self.get_value_from_sheet(self.cdkt_df, "nợ ngắn hạn", column_index=-1)
        hang_ton_kho = self.get_value_from_sheet(self.cdkt_df, "hàng tồn kho", column_index=-1)

        # ✅ THAY ĐỔI: Lấy bình quân hàng tồn kho từ 2 cột cuối
        binh_quan_hang_ton_kho = self.get_average_from_two_periods(self.cdkt_df, "hàng tồn kho")

        # ✅ THAY ĐỔI: Lấy "chi phí Lãi vay" từ LCTT thay vì BCTN
        lai_vay = self.get_value_from_sheet(self.lctt_df, "chi phí lãi vay")
        if lai_vay == 0:
            lai_vay = self.get_value_from_sheet(self.lctt_df, "chi phí lãi")
        if lai_vay == 0:
            lai_vay = self.get_value_from_sheet(self.lctt_df, "lãi vay")

        # ✅ THAY ĐỔI: Lấy "Nợ dài hạn" từ CDKT (thay vì "nợ dài hạn đến hạn")
        no_dai_han = self.get_value_from_sheet(self.cdkt_df, "nợ dài hạn", column_index=-1)

        # ✅ THAY ĐỔI: Lấy "Khấu hao TSCĐ và BĐSĐT" từ LCTT thay vì BCTN
        khau_hao = self.get_value_from_sheet(self.lctt_df, "khấu hao tscđ")
        if khau_hao == 0:
            khau_hao = self.get_value_from_sheet(self.lctt_df, "khấu hao")
        if khau_hao == 0:
            khau_hao = self.get_value_from_sheet(self.lctt_df, "khấu hao tài sản")

        tien_va_tuong_duong = self.get_value_from_sheet(self.cdkt_df, "tiền", column_index=-1)
        if tien_va_tuong_duong == 0:
            tien_va_tuong_duong = self.get_value_from_sheet(self.cdkt_df, "tiền và tương đương", column_index=-1)

        khoan_phai_thu = self.get_value_from_sheet(self.cdkt_df, "phải thu", column_index=-1)
        # ✅ THAY ĐỔI: Lấy bình quân phải thu từ 2 cột cuối
        binh_quan_phai_thu = self.get_average_from_two_periods(self.cdkt_df, "phải thu")

        # Tính 14 chỉ số
        indicators = {}

        # X_1: Hệ số biên lợi nhuận gộp
        indicators['X_1'] = loi_nhuan_gop / doanh_thu_thuan if doanh_thu_thuan != 0 else 0

        # X_2: Hệ số biên lợi nhuận trước thuế
        indicators['X_2'] = loi_nhuan_truoc_thue / doanh_thu_thuan if doanh_thu_thuan != 0 else 0

        # X_3: Tỷ suất lợi nhuận trước thuế trên tổng tài sản (ROA)
        indicators['X_3'] = loi_nhuan_truoc_thue / binh_quan_tong_tai_san if binh_quan_tong_tai_san != 0 else 0

        # X_4: Tỷ suất lợi nhuận trước thuế trên vốn chủ sở hữu (ROE)
        indicators['X_4'] = loi_nhuan_truoc_thue / binh_quan_von_chu_so_huu if binh_quan_von_chu_so_huu != 0 else 0

        # X_5: Hệ số nợ trên tài sản
        indicators['X_5'] = no_phai_tra / tong_tai_san if tong_tai_san != 0 else 0

        # X_6: Hệ số nợ trên vốn chủ sở hữu
        indicators['X_6'] = no_phai_tra / von_chu_so_huu if von_chu_so_huu != 0 else 0

        # X_7: Khả năng thanh toán hiện hành
        indicators['X_7'] = tai_san_ngan_han / no_ngan_han if no_ngan_han != 0 else 0

        # X_8: Khả năng thanh toán nhanh
        indicators['X_8'] = (tai_san_ngan_han - hang_ton_kho) / no_ngan_han if no_ngan_han != 0 else 0

        # X_9: Hệ số khả năng trả lãi
        # ✅ CÔNG THỨC: (Lợi nhuận trước thuế (LCTT) + chi phí Lãi vay (LCTT)) / chi phí Lãi vay (LCTT)
        lntt_cong_lai_vay = loi_nhuan_truoc_thue + lai_vay
        indicators['X_9'] = lntt_cong_lai_vay / lai_vay if lai_vay != 0 else 0

        # X_10: Hệ số khả năng trả nợ gốc
        # ✅ CÔNG THỨC: (LNTT (LCTT) + Lãi vay (LCTT) + Khấu hao (LCTT)) / (Lãi vay (LCTT) + Nợ dài hạn (CDKT))
        tu_so_x10 = lntt_cong_lai_vay + khau_hao
        mau_so_x10 = lai_vay + no_dai_han
        indicators['X_10'] = tu_so_x10 / mau_so_x10 if mau_so_x10 != 0 else 0

        # X_11: Hệ số khả năng tạo tiền trên vốn chủ sở hữu
        indicators['X_11'] = tien_va_tuong_duong / von_chu_so_huu if von_chu_so_huu != 0 else 0

        # X_12: Vòng quay hàng tồn kho
        # ✅ CÔNG THỨC: Giá vốn hàng bán (BCTN) / Bình quân hàng tồn kho (CDKT)
        # ✅ CHUYỂN GIÁ TRỊ ÂM THÀNH DƯƠNG (LẤY GIÁ TRỊ TUYỆT ĐỐI)
        x12_value = gia_von_hang_ban / binh_quan_hang_ton_kho if binh_quan_hang_ton_kho != 0 else 0
        indicators['X_12'] = abs(x12_value)  # Lấy giá trị tuyệt đối (chuyển âm thành dương)

        # X_13: Kỳ thu tiền bình quân
        indicators['X_13'] = 365 / (doanh_thu_thuan / binh_quan_phai_thu) if (doanh_thu_thuan != 0 and binh_quan_phai_thu != 0) else 0

        # X_14: Hiệu suất sử dụng tài sản
        indicators['X_14'] = doanh_thu_thuan / binh_quan_tong_tai_san if binh_quan_tong_tai_san != 0 else 0

        # Làm tròn kết quả
        for key in indicators:
            indicators[key] = round(indicators[key], 6)

        self.financial_indicators = indicators
        return indicators

    def get_indicators_with_names(self) -> Dict[str, Any]:
        """
        Lấy 14 chỉ số kèm tên đầy đủ

        Returns:
            Dict chứa thông tin chi tiết về 14 chỉ số
        """
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
        for key, value in self.financial_indicators.items():
            result.append({
                'code': key,
                'name': indicator_names[key],
                'value': value
            })

        return result

    def simulate_scenario_indicators(
        self,
        original_indicators: Dict[str, float],
        revenue_change_pct: float = 0,
        interest_change_pct: float = 0,
        roe_change_pct: float = 0,
        cr_change_pct: float = 0
    ) -> Dict[str, float]:
        """
        Mô phỏng kịch bản biến động kinh tế và tính lại 14 chỉ số

        Args:
            original_indicators: Dict chứa 14 chỉ số ban đầu (X_1 -> X_14)
            revenue_change_pct: % thay đổi doanh thu thuần (âm = giảm, dương = tăng)
            interest_change_pct: % thay đổi chi phí lãi vay (âm = giảm, dương = tăng)
            roe_change_pct: % thay đổi ROE (âm = giảm, dương = tăng)
            cr_change_pct: % thay đổi Current Ratio (âm = giảm, dương = tăng)

        Returns:
            Dict chứa 14 chỉ số sau khi áp dụng kịch bản biến động

        Logic:
            - Giá trị mới = Giá trị cũ * (1 + %thay đổi)
            - %thay đổi âm = giảm, dương = tăng
            - VD: revenue_change_pct = -5 nghĩa là giảm 5%
                  => Giá trị mới = Giá trị cũ * (1 - 0.05) = Giá trị cũ * 0.95
        """
        import copy
        new_indicators = copy.deepcopy(original_indicators)

        # 1. ẢNH HƯỞNG CỦA DOANH THU THUẦN GIẢM
        # Doanh thu thuần ảnh hưởng trực tiếp đến: X_1, X_2, X_13, X_14
        if revenue_change_pct != 0:
            multiplier = 1 + (revenue_change_pct / 100)

            # X_1: Biên LN gộp = LN gộp / Doanh thu thuần
            # Giả định: LN gộp giảm theo tỷ lệ tương tự doanh thu
            new_indicators['X_1'] = original_indicators['X_1'] * multiplier

            # X_2: Biên LN trước thuế = LN trước thuế / Doanh thu thuần
            # Giả định: LN trước thuế giảm mạnh hơn doanh thu (do chi phí cố định)
            # Hệ số nhân = multiplier ^ 1.2 (ảnh hưởng mạnh hơn)
            new_indicators['X_2'] = original_indicators['X_2'] * (multiplier ** 1.2)

            # X_13: Kỳ thu tiền bình quân = 365 / (Doanh thu / BQ Phải thu)
            # Doanh thu giảm -> Kỳ thu tiền tăng (nghịch đảo)
            new_indicators['X_13'] = original_indicators['X_13'] / multiplier

            # X_14: Hiệu suất tài sản = Doanh thu / BQ Tài sản
            # Doanh thu giảm -> X_14 giảm
            new_indicators['X_14'] = original_indicators['X_14'] * multiplier

            # X_3: ROA = LN trước thuế / BQ Tài sản
            # Giả định: LN trước thuế giảm theo X_2
            new_indicators['X_3'] = original_indicators['X_3'] * (multiplier ** 1.2)

        # 2. ẢNH HƯỞNG CỦA CHI PHÍ LÃI VAY TĂNG
        # Chi phí lãi vay ảnh hưởng đến: X_9, X_10
        if interest_change_pct != 0:
            interest_multiplier = 1 + (interest_change_pct / 100)

            # X_9: Khả năng trả lãi = (LNTT + Lãi vay) / Lãi vay
            # Lãi vay tăng -> X_9 giảm
            # Công thức đảo: new_X9 = (original_X9 * Lãi_cũ - Lãi_cũ + Lãi_mới) / Lãi_mới
            # Đơn giản hóa: X_9 mới ≈ X_9 cũ / interest_multiplier (xấp xỉ)
            new_indicators['X_9'] = original_indicators['X_9'] / (interest_multiplier ** 0.8)

            # X_10: Khả năng trả nợ gốc = (LNTT + Lãi + Khấu hao) / (Lãi + Nợ dài hạn)
            # Lãi vay tăng -> X_10 giảm (nhưng ít hơn X_9)
            new_indicators['X_10'] = original_indicators['X_10'] / (interest_multiplier ** 0.5)

        # 3. ẢNH HƯỞNG CỦA ROE GIẢM
        # ROE (X_4) giảm trực tiếp
        if roe_change_pct != 0:
            roe_multiplier = 1 + (roe_change_pct / 100)
            new_indicators['X_4'] = original_indicators['X_4'] * roe_multiplier

            # ROE giảm cũng ảnh hưởng đến các chỉ số liên quan đến lợi nhuận
            # X_11: Khả năng tạo tiền / VCSH (gián tiếp ảnh hưởng)
            new_indicators['X_11'] = original_indicators['X_11'] * (roe_multiplier ** 0.5)

        # 4. ẢNH HƯỞNG CỦA CURRENT RATIO (CR - X_7) GIẢM
        if cr_change_pct != 0:
            cr_multiplier = 1 + (cr_change_pct / 100)

            # X_7: CR = Tài sản ngắn hạn / Nợ ngắn hạn
            new_indicators['X_7'] = original_indicators['X_7'] * cr_multiplier

            # X_8: Khả năng thanh toán nhanh = (TSNH - HTK) / Nợ NH
            # CR giảm -> X_8 cũng giảm theo
            new_indicators['X_8'] = original_indicators['X_8'] * cr_multiplier

            # X_12: Vòng quay HTK (gián tiếp ảnh hưởng - HTK tăng nếu CR giảm)
            # CR giảm có thể do HTK tăng -> Vòng quay giảm
            new_indicators['X_12'] = original_indicators['X_12'] * (cr_multiplier ** 0.3)

        # 5. CÁC CHỈ SỐ ÍT BỊ ẢNH HƯỞNG (nhưng vẫn có thể biến động nhẹ)
        # X_5, X_6: Tỷ lệ nợ (ít thay đổi trong ngắn hạn)
        # Giữ nguyên hoặc biến động rất nhẹ

        # Làm tròn kết quả
        for key in new_indicators:
            new_indicators[key] = round(new_indicators[key], 6)

        return new_indicators

    def simulate_scenario_full_propagation(
        self,
        original_indicators: Dict[str, float],
        revenue_change_pct: float = 0,
        interest_rate_change_pct: float = 0,
        cogs_change_pct: float = 0,
        liquidity_shock_pct: float = 0
    ) -> Dict[str, float]:
        """
        Mô phỏng kịch bản Stress Testing với tính toán dây chuyền hoàn chỉnh (Phương án A)

        Args:
            original_indicators: Dict chứa 14 chỉ số ban đầu (X_1 -> X_14)
            revenue_change_pct: % thay đổi Doanh thu thuần (âm = giảm, dương = tăng)
            interest_rate_change_pct: % thay đổi Lãi suất vay (âm = giảm, dương = tăng)
            cogs_change_pct: % thay đổi Giá vốn hàng bán (âm = giảm, dương = tăng)
            liquidity_shock_pct: % sốc thanh khoản TSNH (âm = giảm, dương = tăng)

        Returns:
            Dict chứa 14 chỉ số sau khi áp dụng kịch bản stress testing

        Quy trình:
            1. Reverse Engineering: Từ 14 chỉ số ban đầu → Tính ngược ra các biến gốc
            2. Áp dụng Shocks: Thay đổi biến gốc theo 4 input
            3. Tính dây chuyền: Cập nhật các biến phụ thuộc
            4. Tính lại 14 chỉ số: Từ các biến gốc mới
        """

        # ================================================================================
        # BƯỚC 1: REVERSE ENGINEERING - Tính ngược các biến gốc từ 14 chỉ số
        # ================================================================================

        # Giả định các giá trị cơ sở (baseline) để reverse engineering
        # Đây là các giá trị "chuẩn hóa" để tính ngược

        # Giả định Doanh thu thuần ban đầu = 1000 (đơn vị triệu VND)
        doanh_thu_thuan_cu = 1000.0

        # Từ X_1: Hệ số biên LN gộp = LN gộp / Doanh thu
        # => LN gộp = X_1 * Doanh thu
        loi_nhuan_gop_cu = original_indicators['X_1'] * doanh_thu_thuan_cu

        # Từ LN gộp = Doanh thu - Giá vốn
        # => Giá vốn = Doanh thu - LN gộp
        gia_von_hang_ban_cu = doanh_thu_thuan_cu - loi_nhuan_gop_cu

        # Từ X_2: Hệ số biên LN trước thuế = LNTT / Doanh thu
        # => LNTT = X_2 * Doanh thu
        loi_nhuan_truoc_thue_cu = original_indicators['X_2'] * doanh_thu_thuan_cu

        # Từ X_14: Hiệu suất tài sản = Doanh thu / BQ Tài sản
        # => BQ Tài sản = Doanh thu / X_14
        binh_quan_tong_tai_san_cu = doanh_thu_thuan_cu / original_indicators['X_14'] if original_indicators['X_14'] != 0 else 1000

        # Giả định Tổng tài sản cuối kỳ ≈ BQ Tài sản (đơn giản hóa)
        tong_tai_san_cu = binh_quan_tong_tai_san_cu

        # Từ X_4: ROE = LNTT / BQ VCSH
        # => BQ VCSH = LNTT / X_4
        binh_quan_von_chu_so_huu_cu = loi_nhuan_truoc_thue_cu / original_indicators['X_4'] if original_indicators['X_4'] != 0 else 500

        # Giả định VCSH cuối kỳ ≈ BQ VCSH (đơn giản hóa)
        von_chu_so_huu_cu = binh_quan_von_chu_so_huu_cu

        # Từ X_5: Hệ số Nợ/TS = Nợ / Tổng TS
        # => Nợ = X_5 * Tổng TS
        no_phai_tra_cu = original_indicators['X_5'] * tong_tai_san_cu

        # Từ X_7: CR = TSNH / Nợ NH
        # Giả định Nợ NH ≈ 50% Nợ phải trả
        no_ngan_han_cu = no_phai_tra_cu * 0.5

        # => TSNH = X_7 * Nợ NH
        tai_san_ngan_han_cu = original_indicators['X_7'] * no_ngan_han_cu

        # Từ X_8: Khả năng TT nhanh = (TSNH - HTK) / Nợ NH
        # => HTK = TSNH - (X_8 * Nợ NH)
        hang_ton_kho_cu = tai_san_ngan_han_cu - (original_indicators['X_8'] * no_ngan_han_cu)

        # Giả định BQ HTK ≈ HTK cuối kỳ
        binh_quan_hang_ton_kho_cu = hang_ton_kho_cu

        # Từ X_13: Kỳ thu tiền BQ = 365 / (Doanh thu / BQ Phải thu)
        # => BQ Phải thu = 365 * Doanh thu / (X_13 * Doanh thu) = 365 / X_13 * Doanh thu / Doanh thu
        # Đơn giản: BQ Phải thu = Doanh thu * X_13 / 365
        binh_quan_phai_thu_cu = (doanh_thu_thuan_cu * original_indicators['X_13'] / 365) if original_indicators['X_13'] != 0 else 50

        # Từ X_11: Khả năng tạo tiền / VCSH = Tiền / VCSH
        # => Tiền = X_11 * VCSH
        tien_va_tuong_duong_cu = original_indicators['X_11'] * von_chu_so_huu_cu

        # Từ LNTT = LN gộp - Chi phí HĐ - Lãi vay
        # Chi phí HĐ cố định = LN gộp - LNTT - Lãi vay
        # Giả định Lãi vay dựa trên X_9: Khả năng trả lãi = (LNTT + Lãi vay) / Lãi vay
        # => X_9 * Lãi vay = LNTT + Lãi vay
        # => Lãi vay = LNTT / (X_9 - 1)
        lai_vay_cu = loi_nhuan_truoc_thue_cu / (original_indicators['X_9'] - 1) if original_indicators['X_9'] > 1 else 10

        # Chi phí hoạt động cố định = LN gộp - LNTT - Lãi vay
        chi_phi_hoat_dong_co_dinh = max(0, loi_nhuan_gop_cu - loi_nhuan_truoc_thue_cu - lai_vay_cu)

        # Từ X_10: Khả năng trả nợ gốc = (LNTT + Lãi vay + Khấu hao) / (Lãi vay + Nợ DH)
        # => Nợ DH = [(LNTT + Lãi vay + Khấu hao) / X_10] - Lãi vay
        # Giả định Khấu hao ≈ 5% Tổng TS
        khau_hao_cu = tong_tai_san_cu * 0.05

        tu_so_x10 = loi_nhuan_truoc_thue_cu + lai_vay_cu + khau_hao_cu
        no_dai_han_cu = (tu_so_x10 / original_indicators['X_10'] - lai_vay_cu) if original_indicators['X_10'] != 0 else 100

        # ================================================================================
        # BƯỚC 2: ÁP DỤNG SHOCKS - Thay đổi biến gốc theo 4 input
        # ================================================================================

        # Shock 1: Doanh thu thay đổi
        doanh_thu_thuan_moi = doanh_thu_thuan_cu * (1 + revenue_change_pct / 100)

        # Shock 2: Giá vốn thay đổi
        gia_von_hang_ban_moi = gia_von_hang_ban_cu * (1 + cogs_change_pct / 100)

        # Shock 3: Lãi suất vay thay đổi
        lai_vay_moi = lai_vay_cu * (1 + interest_rate_change_pct / 100)

        # Shock 4: Thanh khoản TSNH thay đổi
        tai_san_ngan_han_moi = tai_san_ngan_han_cu * (1 + liquidity_shock_pct / 100)

        # ================================================================================
        # BƯỚC 3: TÍNH DÂY CHUYỀN - Cập nhật các biến phụ thuộc
        # ================================================================================

        # 3.1. Lợi nhuận gộp mới = Doanh thu mới - Giá vốn mới
        loi_nhuan_gop_moi = doanh_thu_thuan_moi - gia_von_hang_ban_moi

        # 3.2. Lợi nhuận trước thuế mới = LN gộp mới - Chi phí HĐ cố định - Lãi vay mới
        # Giả định: Chi phí HĐ cố định không đổi trong ngắn hạn
        loi_nhuan_truoc_thue_moi = loi_nhuan_gop_moi - chi_phi_hoat_dong_co_dinh - lai_vay_moi

        # 3.3. Vốn chủ sở hữu mới = VCSH cũ + (LNTT mới - LNTT cũ)
        # Giả định: Lợi nhuận được giữ lại (không chia cổ tức)
        von_chu_so_huu_moi = von_chu_so_huu_cu + (loi_nhuan_truoc_thue_moi - loi_nhuan_truoc_thue_cu)

        # Đảm bảo VCSH không âm
        von_chu_so_huu_moi = max(50, von_chu_so_huu_moi)

        # 3.4. Nợ phải trả mới = Nợ cũ + vay thêm (nếu lỗ)
        # Nếu LNTT < 0 thì doanh nghiệp cần vay thêm để bù đắp lỗ
        if loi_nhuan_truoc_thue_moi < 0:
            no_phai_tra_moi = no_phai_tra_cu + abs(loi_nhuan_truoc_thue_moi) * 0.5
        else:
            no_phai_tra_moi = no_phai_tra_cu

        # 3.5. Tổng tài sản mới = VCSH mới + Nợ mới
        tong_tai_san_moi = von_chu_so_huu_moi + no_phai_tra_moi

        # 3.6. Hàng tồn kho mới
        # Nếu doanh thu giảm → Bán chậm → HTK tăng
        # HTK mới = HTK cũ × (1 - revenue_change_pct/200)
        # Chia 200 để ảnh hưởng nhẹ hơn (50% của revenue change)
        hang_ton_kho_moi = hang_ton_kho_cu * (1 - revenue_change_pct / 200)
        hang_ton_kho_moi = max(0, hang_ton_kho_moi)

        # 3.7. Nợ ngắn hạn mới
        # Nếu doanh thu giảm → Cần vay ngắn hạn để duy trì hoạt động
        # NNH mới = NNH cũ × (1 - revenue_change_pct/200)
        no_ngan_han_moi = no_ngan_han_cu * (1 - revenue_change_pct / 200)
        no_ngan_han_moi = max(50, no_ngan_han_moi)

        # 3.8. Tiền và tương đương tiền mới
        # Bị ảnh hưởng bởi thanh khoản và lợi nhuận
        tien_va_tuong_duong_moi = tien_va_tuong_duong_cu * (1 + liquidity_shock_pct / 100)
        # Nếu lỗ thì tiền giảm thêm
        if loi_nhuan_truoc_thue_moi < 0:
            tien_va_tuong_duong_moi = max(10, tien_va_tuong_duong_moi + loi_nhuan_truoc_thue_moi * 0.3)
        tien_va_tuong_duong_moi = max(10, tien_va_tuong_duong_moi)

        # 3.9. Phải thu bình quân mới
        # Phải thu tăng nếu doanh thu giảm (khách hàng trả chậm)
        binh_quan_phai_thu_moi = binh_quan_phai_thu_cu * (1 - revenue_change_pct / 150)
        binh_quan_phai_thu_moi = max(10, binh_quan_phai_thu_moi)

        # 3.10. Bình quân tổng tài sản mới
        # Giả định BQ TS ≈ TS cuối kỳ (đơn giản hóa)
        binh_quan_tong_tai_san_moi = tong_tai_san_moi

        # 3.11. Bình quân VCSH mới
        binh_quan_von_chu_so_huu_moi = von_chu_so_huu_moi

        # 3.12. Bình quân HTK mới
        binh_quan_hang_ton_kho_moi = hang_ton_kho_moi

        # 3.13. Khấu hao mới (giả định không đổi hoặc theo TS mới)
        khau_hao_moi = tong_tai_san_moi * 0.05

        # 3.14. Nợ dài hạn mới (giả định không đổi trong ngắn hạn)
        no_dai_han_moi = no_dai_han_cu

        # ================================================================================
        # BƯỚC 4: TÍNH LẠI 14 CHỈ SỐ - Từ các biến gốc mới
        # ================================================================================

        new_indicators = {}

        # X_1: Hệ số biên lợi nhuận gộp
        new_indicators['X_1'] = loi_nhuan_gop_moi / doanh_thu_thuan_moi if doanh_thu_thuan_moi != 0 else 0

        # X_2: Hệ số biên lợi nhuận trước thuế
        new_indicators['X_2'] = loi_nhuan_truoc_thue_moi / doanh_thu_thuan_moi if doanh_thu_thuan_moi != 0 else 0

        # X_3: Tỷ suất lợi nhuận trước thuế trên tổng tài sản (ROA)
        new_indicators['X_3'] = loi_nhuan_truoc_thue_moi / binh_quan_tong_tai_san_moi if binh_quan_tong_tai_san_moi != 0 else 0

        # X_4: Tỷ suất lợi nhuận trước thuế trên vốn chủ sở hữu (ROE)
        new_indicators['X_4'] = loi_nhuan_truoc_thue_moi / binh_quan_von_chu_so_huu_moi if binh_quan_von_chu_so_huu_moi != 0 else 0

        # X_5: Hệ số nợ trên tài sản
        new_indicators['X_5'] = no_phai_tra_moi / tong_tai_san_moi if tong_tai_san_moi != 0 else 0

        # X_6: Hệ số nợ trên vốn chủ sở hữu
        new_indicators['X_6'] = no_phai_tra_moi / von_chu_so_huu_moi if von_chu_so_huu_moi != 0 else 0

        # X_7: Khả năng thanh toán hiện hành
        new_indicators['X_7'] = tai_san_ngan_han_moi / no_ngan_han_moi if no_ngan_han_moi != 0 else 0

        # X_8: Khả năng thanh toán nhanh
        new_indicators['X_8'] = (tai_san_ngan_han_moi - hang_ton_kho_moi) / no_ngan_han_moi if no_ngan_han_moi != 0 else 0

        # X_9: Hệ số khả năng trả lãi
        lntt_cong_lai_vay_moi = loi_nhuan_truoc_thue_moi + lai_vay_moi
        new_indicators['X_9'] = lntt_cong_lai_vay_moi / lai_vay_moi if lai_vay_moi != 0 else 0

        # X_10: Hệ số khả năng trả nợ gốc
        tu_so_x10_moi = lntt_cong_lai_vay_moi + khau_hao_moi
        mau_so_x10_moi = lai_vay_moi + no_dai_han_moi
        new_indicators['X_10'] = tu_so_x10_moi / mau_so_x10_moi if mau_so_x10_moi != 0 else 0

        # X_11: Hệ số khả năng tạo tiền trên vốn chủ sở hữu
        new_indicators['X_11'] = tien_va_tuong_duong_moi / von_chu_so_huu_moi if von_chu_so_huu_moi != 0 else 0

        # X_12: Vòng quay hàng tồn kho
        x12_value = gia_von_hang_ban_moi / binh_quan_hang_ton_kho_moi if binh_quan_hang_ton_kho_moi != 0 else 0
        new_indicators['X_12'] = abs(x12_value)  # Lấy giá trị tuyệt đối

        # X_13: Kỳ thu tiền bình quân
        new_indicators['X_13'] = 365 / (doanh_thu_thuan_moi / binh_quan_phai_thu_moi) if (doanh_thu_thuan_moi != 0 and binh_quan_phai_thu_moi != 0) else 0

        # X_14: Hiệu suất sử dụng tài sản
        new_indicators['X_14'] = doanh_thu_thuan_moi / binh_quan_tong_tai_san_moi if binh_quan_tong_tai_san_moi != 0 else 0

        # Làm tròn kết quả
        for key in new_indicators:
            new_indicators[key] = round(new_indicators[key], 6)

        return new_indicators

    def macro_to_micro_transmission(
        self,
        gdp_growth_pct: float,
        inflation_cpi_pct: float,
        inflation_ppi_pct: float,
        policy_rate_change_bps: float,
        fx_usd_vnd_pct: float,
        industry_code: str
    ) -> Dict[str, float]:
        """
        Kênh truyền dẫn từ biến vĩ mô sang biến vi mô (Macro-to-Micro Transmission)

        Args:
            gdp_growth_pct: % tăng trưởng GDP (VD: -3.5 = giảm 3.5%)
            inflation_cpi_pct: % lạm phát CPI (VD: 10.0 = lạm phát 10%)
            inflation_ppi_pct: % lạm phát PPI - giá sản xuất (VD: 14.0)
            policy_rate_change_bps: Thay đổi lãi suất NHNN (basis points, VD: 200 = tăng 2%)
            fx_usd_vnd_pct: % thay đổi tỷ giá USD/VND (VD: 6.0 = VND mất giá 6%)
            industry_code: Mã ngành ("manufacturing", "export", "retail")

        Returns:
            Dict chứa 4 biến vi mô:
            - revenue_change_pct: % thay đổi Doanh thu thuần
            - cogs_change_pct: % thay đổi Giá vốn hàng bán
            - interest_rate_change_pct: % thay đổi Lãi suất vay
            - liquidity_shock_pct: % sốc thanh khoản TSNH

        Công thức kênh truyền dẫn:
            1. GDP → Doanh thu:
               revenue_change = (GDP * 0.8 + CPI * 0.2) * industry_sensitivity["revenue"]

            2. PPI + Tỷ giá → Giá vốn:
               cogs_change = (PPI * 0.7 + FX * 0.3) * industry_sensitivity["cogs"]

            3. Lãi suất NHNN → Lãi vay:
               interest_rate_change = policy_rate_bps / 100 * 1.2

            4. GDP + Lãi suất → Thanh khoản:
               liquidity_shock = GDP * 0.5 + policy_rate_bps / 100 * (-0.8)
        """

        # Hệ số nhạy cảm ngành (Industry Sensitivity)
        industry_sensitivity = {
            "manufacturing": {  # Sản xuất
                "revenue": 1.0,
                "cogs": 1.2
            },
            "export": {  # Xuất khẩu
                "revenue": 1.3,
                "cogs": 1.1
            },
            "retail": {  # Bán lẻ
                "revenue": 0.8,
                "cogs": 0.9
            }
        }

        # Lấy hệ số ngành (mặc định là manufacturing nếu không tìm thấy)
        sensitivity = industry_sensitivity.get(industry_code, industry_sensitivity["manufacturing"])

        # ================================================================================
        # KÊNH 1: GDP → Doanh thu thuần
        # GDP tăng → Tiêu dùng tăng → Doanh thu tăng
        # CPI tăng → Sức mua giảm → Doanh thu giảm (trọng số nhỏ hơn)
        # ================================================================================
        revenue_change_pct = (
            gdp_growth_pct * 0.8 +
            inflation_cpi_pct * 0.2
        ) * sensitivity["revenue"]

        # ================================================================================
        # KÊNH 2: PPI + Tỷ giá → Giá vốn hàng bán
        # PPI tăng → Giá nguyên liệu tăng → Giá vốn tăng
        # Tỷ giá tăng (VND mất giá) → Nhập khẩu nguyên liệu đắt hơn → Giá vốn tăng
        # ================================================================================
        cogs_change_pct = (
            inflation_ppi_pct * 0.7 +
            fx_usd_vnd_pct * 0.3
        ) * sensitivity["cogs"]

        # ================================================================================
        # KÊNH 3: Lãi suất NHNN → Lãi suất vay doanh nghiệp
        # NHNN tăng lãi suất → Ngân hàng tăng lãi suất cho vay
        # Hệ số nhân 1.2: Lãi suất cho vay thường tăng mạnh hơn lãi suất NHNN
        # ================================================================================
        # Chuyển từ basis points sang % (100 bps = 1%)
        interest_rate_change_pct = (policy_rate_change_bps / 100) * 1.2

        # ================================================================================
        # KÊNH 4: GDP + Lãi suất → Thanh khoản (TSNH)
        # GDP giảm → Doanh thu giảm → Thu hồi tiền chậm → Thanh khoản giảm
        # Lãi suất tăng → Vay khó hơn → Thanh khoản giảm
        # ================================================================================
        liquidity_shock_pct = (
            gdp_growth_pct * 0.5 +
            (policy_rate_change_bps / 100) * (-0.8)
        )

        # Làm tròn kết quả
        result = {
            "revenue_change_pct": round(revenue_change_pct, 2),
            "cogs_change_pct": round(cogs_change_pct, 2),
            "interest_rate_change_pct": round(interest_rate_change_pct, 2),
            "liquidity_shock_pct": round(liquidity_shock_pct, 2)
        }

        print(f"📊 Kênh truyền dẫn Macro → Micro:")
        print(f"   - Doanh thu thay đổi: {result['revenue_change_pct']}%")
        print(f"   - Giá vốn thay đổi: {result['cogs_change_pct']}%")
        print(f"   - Lãi suất vay thay đổi: {result['interest_rate_change_pct']}%")
        print(f"   - Thanh khoản sốc: {result['liquidity_shock_pct']}%")

        return result


# Khởi tạo instance global
excel_processor = ExcelProcessor()
