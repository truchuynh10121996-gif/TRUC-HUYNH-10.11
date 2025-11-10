<template>
  <div>
    <h3 class="chart-title">
      📊 Biểu đồ Phân tích 14 Chỉ số Tài chính
    </h3>

    <!-- Grid 2 biểu đồ nằm ngang -->
    <div class="charts-horizontal-grid">
      <!-- Biểu đồ Cột - Tất cả 14 chỉ số -->
      <div class="chart-wrapper-compact">
        <h4 class="chart-subtitle-compact">📈 Biểu đồ Cột - Toàn bộ 14 Chỉ số</h4>
        <div class="chart-canvas-compact">
          <Bar :data="allIndicatorsBarData" :options="allIndicatorsBarOptions" />
        </div>
      </div>

      <!-- Biểu đồ Radar - Tất cả 14 chỉ số -->
      <div class="chart-wrapper-compact">
        <h4 class="chart-subtitle-compact">🎯 Biểu đồ Radar - Tổng quan 14 Chỉ số</h4>
        <div class="chart-canvas-compact">
          <Radar :data="radarAllData" :options="radarAllOptions" />
        </div>
      </div>
    </div>

    <!-- Bảng note hướng dẫn xem biểu đồ -->
    <div class="chart-note-guide">
      <div class="note-title">📌 Hướng dẫn xem biểu đồ:</div>
      <ul class="note-list">
        <li><strong>Biểu đồ Cột:</strong> Thể hiện giá trị tuyệt đối của từng chỉ số, dễ so sánh độ lớn giữa các chỉ số</li>
        <li><strong>Biểu đồ Radar:</strong> Cho cái nhìn tổng quan toàn diện về 14 chỉ số, giúp phát hiện điểm mạnh/yếu của doanh nghiệp</li>
        <li><strong>Màu sắc:</strong> Mỗi chỉ số có màu riêng biệt để dễ nhận diện</li>
      </ul>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { Bar, Radar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler
} from 'chart.js'

ChartJS.register(
  Title,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler
)

export default {
  name: 'IndicatorsChart',
  components: {
    Bar,
    Radar
  },
  props: {
    indicators: {
      type: Object,
      required: true
    }
  },
  setup(props) {
    // Màu pastel đẹp cho tất cả 14 chỉ số
    const all14Colors = [
      '#FFB3D9', '#FFC4E5', '#FFD1EC', '#FFE0F5', // X1-X4: Màu hồng pastel
      '#FFE0B2', '#FFCC80', // X5-X6: Màu cam pastel
      '#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A', '#4CAF50', // X7-X11: Màu xanh lá pastel
      '#B39DDB', '#9575CD', '#7E57C2' // X12-X14: Màu tím pastel
    ]

    const all14BorderColors = [
      '#FF6B9D', '#FF8AB5', '#FFA8D3', '#FFC4E5',
      '#FF9800', '#F57C00',
      '#66BB6A', '#4CAF50', '#388E3C', '#2E7D32', '#1B5E20',
      '#7E57C2', '#673AB7', '#5E35B1'
    ]

    // Biểu đồ Cột cho tất cả 14 chỉ số
    const allIndicatorsBarData = computed(() => {
      const labels = [
        'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7',
        'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14'
      ]
      const values = []
      for (let i = 1; i <= 14; i++) {
        values.push(props.indicators[`X_${i}`] || 0)
      }

      return {
        labels: labels,
        datasets: [{
          label: 'Giá trị chỉ số',
          data: values,
          backgroundColor: all14Colors,
          borderColor: all14BorderColors,
          borderWidth: 2,
          borderRadius: 6
        }]
      }
    })

    const allIndicatorsBarOptions = {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.5,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          padding: 10,
          cornerRadius: 8,
          titleFont: { size: 11, weight: 'bold' },
          bodyFont: { size: 10 },
          callbacks: {
            label: (context) => `Giá trị: ${context.parsed.y.toFixed(4)}`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            font: { size: 9, weight: 'bold' },
            color: '#4A4A4A'
          },
          grid: { color: 'rgba(255, 182, 193, 0.2)' }
        },
        x: {
          ticks: {
            font: { size: 9 },
            color: '#4A4A4A'
          },
          grid: { display: false }
        }
      }
    }

    // Biểu đồ Radar cho tất cả 14 chỉ số
    const radarAllData = computed(() => {
      const labels = [
        'X1: Biên LN gộp',
        'X2: Biên LN trước thuế',
        'X3: ROA',
        'X4: ROE',
        'X5: Nợ/TS',
        'X6: Nợ/VCSH',
        'X7: TT hiện hành',
        'X8: TT nhanh',
        'X9: KN trả lãi',
        'X10: KN trả nợ',
        'X11: Tiền/VCSH',
        'X12: Vòng quay HTK',
        'X13: Kỳ thu tiền',
        'X14: Hiệu suất TS'
      ]

      const values = []
      for (let i = 1; i <= 14; i++) {
        values.push(props.indicators[`X_${i}`] || 0)
      }

      return {
        labels: labels,
        datasets: [{
          label: '14 Chỉ số Tài chính',
          data: values,
          backgroundColor: 'rgba(255, 107, 157, 0.25)',
          borderColor: '#FF6B9D',
          borderWidth: 3,
          pointBackgroundColor: '#FF6B9D',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#FF6B9D',
          pointRadius: 5,
          pointHoverRadius: 7
        }]
      }
    })

    const radarAllOptions = {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.5,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          titleFont: { size: 11, weight: 'bold' },
          bodyFont: { size: 10 },
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: (context) => {
              const value = context.parsed.r
              return `Giá trị: ${value.toFixed(4)}`
            }
          }
        }
      },
      scales: {
        r: {
          beginAtZero: true,
          ticks: {
            stepSize: 0.2,
            font: { size: 8, weight: 'bold' },
            color: '#4A4A4A',
            backdropColor: 'rgba(255, 255, 255, 0.8)',
            backdropPadding: 2
          },
          grid: {
            color: 'rgba(255, 182, 193, 0.4)',
            lineWidth: 1.5
          },
          angleLines: {
            color: 'rgba(255, 182, 193, 0.4)',
            lineWidth: 1.5
          },
          pointLabels: {
            font: { size: 8, weight: 'bold' },
            color: '#2c3e50',
            padding: 5
          }
        }
      }
    }

    return {
      allIndicatorsBarData,
      allIndicatorsBarOptions,
      radarAllData,
      radarAllOptions
    }
  }
}
</script>

<style scoped>
.chart-title {
  font-size: 1.3rem;
  font-weight: 700;
  color: #FF6B9D;
  text-align: center;
  margin-bottom: 1.5rem;
  text-shadow: 1px 1px 2px rgba(255, 182, 193, 0.3);
}

/* Grid ngang cho 2 biểu đồ - kích thước giảm 30% */
.charts-horizontal-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin: 1.5rem 0;
}

@media (max-width: 1024px) {
  .charts-horizontal-grid {
    grid-template-columns: 1fr;
  }
}

/* Wrapper cho mỗi biểu đồ - compact và đẹp */
.chart-wrapper-compact {
  background: linear-gradient(135deg,
    rgba(255, 255, 255, 0.98) 0%,
    rgba(255, 245, 250, 0.98) 100%);
  border-radius: 16px;
  padding: 1rem;
  box-shadow: 0 3px 12px rgba(255, 182, 193, 0.25);
  border: 1.5px solid rgba(255, 182, 193, 0.3);
  transition: all 0.3s ease;
}

.chart-wrapper-compact:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 18px rgba(255, 182, 193, 0.35);
  border-color: rgba(255, 107, 157, 0.5);
}

.chart-subtitle-compact {
  font-size: 0.9rem;
  font-weight: 600;
  color: #FF6B9D;
  margin-bottom: 0.8rem;
  text-align: center;
  padding-bottom: 0.5rem;
  border-bottom: 1.5px solid rgba(255, 182, 193, 0.3);
}

/* Canvas container - giảm kích thước */
.chart-canvas-compact {
  width: 100%;
  height: auto;
  max-height: 280px;
}

/* Bảng note hướng dẫn - nhỏ gọn */
.chart-note-guide {
  background: linear-gradient(135deg,
    rgba(255, 250, 240, 0.95) 0%,
    rgba(255, 245, 250, 0.95) 100%);
  border-radius: 10px;
  padding: 0.8rem 1rem;
  margin-top: 1rem;
  border: 1.5px dashed rgba(255, 182, 193, 0.4);
  box-shadow: 0 2px 8px rgba(255, 182, 193, 0.15);
}

.note-title {
  font-size: 0.85rem;
  font-weight: 700;
  color: #FF6B9D;
  margin-bottom: 0.5rem;
}

.note-list {
  margin: 0;
  padding-left: 1.2rem;
  list-style: disc;
}

.note-list li {
  font-size: 0.75rem;
  color: #4A4A4A;
  line-height: 1.5;
  margin-bottom: 0.3rem;
}

.note-list li:last-child {
  margin-bottom: 0;
}

.note-list strong {
  color: #FF6B9D;
  font-weight: 600;
}
</style>
