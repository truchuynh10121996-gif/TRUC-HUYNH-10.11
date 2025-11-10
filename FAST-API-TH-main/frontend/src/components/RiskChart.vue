<template>
  <div>
    <h3 class="chart-main-title">
      📊 Biểu đồ So sánh Xác suất Vỡ nợ (PD) từ 4 Models
    </h3>

    <!-- Container nằm ngang: Biểu đồ bên trái, Hướng dẫn bên phải -->
    <div class="chart-horizontal-layout">
      <!-- Biểu đồ cột - chiếm 50% -->
      <div class="chart-container-compact">
        <Bar :data="chartData" :options="chartOptions" />
      </div>

      <!-- Hướng dẫn đọc - chiếm 50% -->
      <div class="chart-guide-compact">
        <div class="guide-title">📖 Cách đọc biểu đồ</div>
        <ul class="guide-list">
          <li>Biểu đồ cột so sánh xác suất vỡ nợ (PD) từ 4 mô hình AI</li>
          <li><span class="color-dot green">●</span> <strong>Xanh:</strong> Rủi ro thấp (&lt;5%)</li>
          <li><span class="color-dot yellow">●</span> <strong>Vàng:</strong> Rủi ro trung bình (5-15%)</li>
          <li><span class="color-dot red">●</span> <strong>Đỏ:</strong> Rủi ro cao (&gt;15%)</li>
          <li class="guide-highlight">⭐ <strong>Stacking Model</strong> là kết quả tổng hợp tin cậy nhất</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale
} from 'chart.js'

ChartJS.register(
  Title,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  LinearScale
)

export default {
  name: 'RiskChart',
  components: {
    Bar
  },
  props: {
    prediction: {
      type: Object,
      required: true
    }
  },
  setup(props) {
    const chartData = computed(() => {
      const pdStacking = (props.prediction.pd_stacking * 100).toFixed(2)
      const pdLogistic = (props.prediction.pd_logistic * 100).toFixed(2)
      const pdRF = (props.prediction.pd_random_forest * 100).toFixed(2)
      const pdXGB = (props.prediction.pd_xgboost * 100).toFixed(2)

      return {
        labels: ['Stacking', 'Logistic', 'Random Forest', 'XGBoost'],
        datasets: [
          {
            label: 'Xác suất Vỡ nợ (%)',
            data: [pdStacking, pdLogistic, pdRF, pdXGB],
            backgroundColor: [
              getBarColor(pdStacking / 100),
              getBarColor(pdLogistic / 100),
              getBarColor(pdRF / 100),
              getBarColor(pdXGB / 100)
            ],
            borderColor: [
              getBarBorderColor(pdStacking / 100),
              getBarBorderColor(pdLogistic / 100),
              getBarBorderColor(pdRF / 100),
              getBarBorderColor(pdXGB / 100)
            ],
            borderWidth: 3,
            borderRadius: 10
          }
        ]
      }
    })

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.3,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            font: {
              size: 15,
              weight: 'bold'
            },
            color: '#FF6B9D',
            padding: 20
          }
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          titleFont: {
            size: 15,
            weight: 'bold'
          },
          bodyFont: {
            size: 14
          },
          padding: 15,
          cornerRadius: 10,
          callbacks: {
            label: function(context) {
              const value = context.parsed.y
              let risk = ''
              if (value < 2) risk = '🟢 Rủi ro Rất Thấp (AAA-AA)'
              else if (value < 5) risk = '🟢 Rủi ro Thấp (A-BBB)'
              else if (value < 10) risk = '🟡 Rủi ro Trung bình (BB)'
              else if (value < 20) risk = '🟠 Rủi ro Cao (B)'
              else risk = '🔴 Rủi ro Rất Cao (CCC-D)'
              return `PD: ${value}% - ${risk}`
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: {
            display: true,
            text: 'Xác suất Vỡ nợ (%)',
            font: {
              size: 14,
              weight: 'bold'
            },
            color: '#FF6B9D'
          },
          ticks: {
            callback: function(value) {
              return value + '%'
            },
            font: {
              size: 13,
              weight: 'bold'
            },
            color: '#4A4A4A'
          },
          grid: {
            color: 'rgba(255, 182, 193, 0.2)'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Mô hình Dự báo',
            font: {
              size: 14,
              weight: 'bold'
            },
            color: '#FF6B9D'
          },
          ticks: {
            font: {
              size: 13,
              weight: 'bold'
            },
            color: '#4A4A4A'
          },
          grid: {
            display: false
          }
        }
      }
    }

    const getBarColor = (pd) => {
      if (pd < 0.05) return 'rgba(143, 227, 207, 0.8)' // Xanh nhạt
      if (pd < 0.15) return 'rgba(255, 224, 138, 0.8)' // Vàng nhạt
      return 'rgba(255, 138, 138, 0.8)' // Đỏ nhạt
    }

    const getBarBorderColor = (pd) => {
      if (pd < 0.05) return '#00a651' // Xanh đậm
      if (pd < 0.15) return '#ff9800' // Vàng đậm
      return '#e53935' // Đỏ đậm
    }

    return {
      chartData,
      chartOptions
    }
  }
}
</script>

<style scoped>
.chart-main-title {
  margin-bottom: 1rem;
  color: #FF6B9D;
  text-align: center;
  font-size: 1.2rem;
  font-weight: 700;
}

/* Layout nằm ngang - giảm 50% kích thước */
.chart-horizontal-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  max-width: 900px;
  margin: 0 auto;
  align-items: start;
}

@media (max-width: 968px) {
  .chart-horizontal-layout {
    grid-template-columns: 1fr;
  }
}

/* Container biểu đồ - nhỏ gọn */
.chart-container-compact {
  background: rgba(255, 255, 255, 0.98);
  border-radius: 14px;
  padding: 1rem;
  box-shadow: 0 4px 14px rgba(255, 182, 193, 0.25);
  border: 2px solid rgba(255, 182, 193, 0.3);
}

/* Hướng dẫn - nhỏ gọn, đẹp */
.chart-guide-compact {
  background: linear-gradient(135deg,
    rgba(255, 245, 250, 0.98) 0%,
    rgba(255, 255, 255, 0.98) 100%);
  border-radius: 14px;
  padding: 1rem;
  border: 2px solid rgba(255, 182, 193, 0.3);
  box-shadow: 0 4px 14px rgba(255, 182, 193, 0.25);
}

.guide-title {
  font-size: 0.9rem;
  font-weight: 700;
  color: #FF6B9D;
  margin-bottom: 0.8rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid rgba(255, 182, 193, 0.3);
}

.guide-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.guide-list li {
  font-size: 0.75rem;
  color: #4A4A4A;
  line-height: 1.6;
  margin-bottom: 0.5rem;
  padding-left: 0.3rem;
}

.guide-list li:last-child {
  margin-bottom: 0;
}

.color-dot {
  font-size: 1rem;
  margin-right: 0.3rem;
}

.color-dot.green {
  color: #00a651;
}

.color-dot.yellow {
  color: #ff9800;
}

.color-dot.red {
  color: #e53935;
}

.guide-highlight {
  background: rgba(255, 182, 193, 0.15);
  padding: 0.4rem 0.6rem;
  border-radius: 6px;
  margin-top: 0.5rem;
  border-left: 3px solid #FF6B9D;
}
</style>
