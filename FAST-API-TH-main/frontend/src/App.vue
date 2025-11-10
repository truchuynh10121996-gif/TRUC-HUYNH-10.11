<template>
  <div id="app">
    <!-- Khoảng trống 1cm trước header -->
    <div class="header-spacer"></div>

    <!-- Nút Lên đầu trang -->
    <button
      v-show="showScrollTop"
      @click="scrollToTop"
      class="scroll-to-top"
      :style="{ top: scrollTopPosition + 'px' }"
    >
      ↑
    </button>

    <!-- Header mới với tông màu hồng lung linh - Chỉ thanh hồng -->
    <header class="header"></header>

    <!-- Logo và Tiêu đề nằm dưới header, canh giữa -->
    <div class="logo-title-section">
      <div class="logo-container-center">
        <img
          src="/logo-agribank1.png"
          alt="Agribank Logo"
          class="logo-center"
        />
      </div>
      <div class="title-section-center">
        <h1 class="main-title-center">CHƯƠNG TRÌNH ĐÁNH GIÁ RỦI RO TÍN DỤNG</h1>
        <h2 class="sub-title-center">Dự báo xác suất Vỡ nợ KHDN (PD) & Phân tích AI chuyên sâu</h2>
      </div>
    </div>

    <!-- Divider sau logo và tiêu đề -->
    <div class="title-divider"></div>

    <!-- ✅ TAB SYSTEM - Thay thế Sidebar -->
    <div class="tabs-container">
      <button
        @click="activeTab = 'predict'"
        class="tab-button"
        :class="{ active: activeTab === 'predict' }"
      >
        🔮 Dự Báo PD
      </button>
      <button
        @click="activeTab = 'dashboard'"
        class="tab-button"
        :class="{ active: activeTab === 'dashboard' }"
      >
        📊 Dashboard Tài Chính
      </button>
      <button
        @click="activeTab = 'scenario'"
        class="tab-button"
        :class="{ active: activeTab === 'scenario' }"
      >
        ⚠️ Mô phỏng kịch bản xấu
      </button>
      <button
        @click="activeTab = 'macro'"
        class="tab-button"
        :class="{ active: activeTab === 'macro' }"
      >
        📊 Mô phỏng Vĩ mô
      </button>
      <button
        @click="activeTab = 'train'"
        class="tab-button"
        :class="{ active: activeTab === 'train' }"
      >
        📚 Huấn luyện mô hình
      </button>
      <button
        @click="activeTab = 'early-warning'"
        class="tab-button"
        :class="{ active: activeTab === 'early-warning' }"
        style="background: linear-gradient(135deg, #FF6B6B 0%, #FFB347 100%); color: white; font-weight: 700;"
      >
        ⚠️ Cảnh báo Rủi ro Sớm
      </button>
      <button
        @click="activeTab = 'anomaly'"
        class="tab-button"
        :class="{ active: activeTab === 'anomaly' }"
        style="background: linear-gradient(135deg, #FF4444 0%, #FF8844 100%); color: white; font-weight: 700;"
      >
        🚨 Phát hiện Gian lận
      </button>
    </div>

    <!-- Main Container -->
    <div class="container">
      <!-- ✅ TAB CONTENT: Dự Báo PD -->
      <div v-if="activeTab === 'predict'" class="tab-content">
        <div class="card">
          <h2 class="card-title">🔮 Dự báo PD & Phân tích AI cho Hồ sơ mới</h2>

          <!-- Ghi chú hướng dẫn -->
          <div class="info-note">
            <span class="note-icon">📝</span>
            <span class="note-text">Tải hồ sơ doanh nghiệp gồm 3 phần Cân đối kế toán, Báo cáo thu nhập, Lưu chuyển tiền tệ và Bắt đầu Dự báo bằng mô hình Stacking Ensemble</span>
          </div>

        <!-- Upload XLSX File -->
        <div style="margin-bottom: 2rem;">
          <div class="upload-area" @click="$refs.xlsxFileInput.click()">
            <div class="upload-icon">📊</div>
            <p class="upload-text">{{ xlsxFileName || 'Tải lên file XLSX của doanh nghiệp' }}</p>
            <p class="upload-hint">
              File XLSX phải có 3 sheets: CDKT (Cân đối kế toán), BCTN (Báo cáo thu nhập), LCTT (Lưu chuyển tiền tệ)
            </p>
          </div>
          <input
            ref="xlsxFileInput"
            type="file"
            accept=".xlsx,.xls"
            @change="handleXlsxFile"
            style="display: none"
          />
          <button
            @click="predictFromXlsx"
            class="btn btn-primary"
            :disabled="!xlsxFile || isPredicting"
            style="margin-top: 1rem; width: 100%;"
          >
            {{ isPredicting ? '⏳ Đang tính toán...' : '🎯 Tính toán 14 chỉ số và Dự báo PD' }}
          </button>
        </div>

        <!-- Results Section -->
        <div v-if="predictionResult">
          <!-- 14 Chỉ số tài chính - 2 bảng nằm ngang -->
          <div style="margin: 3rem 0;">
            <h3 style="margin-bottom: 1.5rem; color: #FF6B9D; text-align: center; font-size: 1.6rem;">
              📈 14 Chỉ số Tài chính đã tính toán
            </h3>
            <div class="indicators-tables-container">
              <!-- Bảng 1: X1-X7 -->
              <div class="indicators-table-wrapper">
                <h4 class="table-subtitle">Nhóm 1: Sinh lời & Thanh toán (X1-X7)</h4>
                <table class="indicators-table">
                  <thead>
                    <tr>
                      <th>Chỉ số</th>
                      <th>Giá trị</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="indicator in indicators.slice(0, 7)" :key="indicator.code">
                      <td>
                        <div class="indicator-code-cell">{{ indicator.code }}</div>
                        <div class="indicator-name-cell">{{ indicator.name }}</div>
                      </td>
                      <td class="indicator-value-cell">{{ indicator.value.toFixed(4) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <!-- Bảng 2: X8-X14 -->
              <div class="indicators-table-wrapper">
                <h4 class="table-subtitle">Nhóm 2: Hiệu quả hoạt động (X8-X14)</h4>
                <table class="indicators-table">
                  <thead>
                    <tr>
                      <th>Chỉ số</th>
                      <th>Giá trị</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="indicator in indicators.slice(7, 14)" :key="indicator.code">
                      <td>
                        <div class="indicator-code-cell">{{ indicator.code }}</div>
                        <div class="indicator-name-cell">{{ indicator.name }}</div>
                      </td>
                      <td class="indicator-value-cell">{{ indicator.value.toFixed(4) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- Dashboard Biểu đồ 14 chỉ số -->
          <div style="margin: 3rem 0;">
            <IndicatorsChart v-if="indicatorsDict" :indicators="indicatorsDict" />
          </div>

          <!-- PD Results - 3 mô hình con trước, Stacking nổi bật ở dưới -->
          <div style="margin: 3rem 0;">
            <h3 style="margin-bottom: 1.5rem; color: #FF6B9D; text-align: center; font-size: 1.6rem;">
              🎯 Kết quả Dự báo Xác suất Vỡ nợ (PD)
            </h3>

            <!-- 3 mô hình con -->
            <div style="margin-bottom: 1rem;">
              <h4 style="color: #7A7A7A; font-size: 1.1rem; margin-bottom: 1rem; text-align: center;">
                📊 Kết quả từ 3 Mô hình Cơ sở
              </h4>
              <div class="pd-grid-base-models">
                <div
                  class="pd-card pd-card-base"
                  :class="getRiskClass(predictionResult.pd_logistic)"
                >
                  <div class="pd-label">📈 Logistic Regression</div>
                  <div class="pd-value">{{ (predictionResult.pd_logistic * 100).toFixed(2) }}%</div>
                  <div class="pd-status">{{ getRiskLabel(predictionResult.pd_logistic) }}</div>
                </div>

                <div
                  class="pd-card pd-card-base"
                  :class="getRiskClass(predictionResult.pd_random_forest)"
                >
                  <div class="pd-label">🌳 Random Forest</div>
                  <div class="pd-value">{{ (predictionResult.pd_random_forest * 100).toFixed(2) }}%</div>
                  <div class="pd-status">{{ getRiskLabel(predictionResult.pd_random_forest) }}</div>
                </div>

                <div
                  class="pd-card pd-card-base"
                  :class="getRiskClass(predictionResult.pd_xgboost)"
                >
                  <div class="pd-label">⚡ XGBoost</div>
                  <div class="pd-value">{{ (predictionResult.pd_xgboost * 100).toFixed(2) }}%</div>
                  <div class="pd-status">{{ getRiskLabel(predictionResult.pd_xgboost) }}</div>
                </div>
              </div>
            </div>

            <!-- Stacking - Kết quả chính nổi bật -->
            <div style="margin-top: 2.5rem;">
              <h4 style="color: #FF6B9D; font-size: 1.3rem; margin-bottom: 1rem; text-align: center; font-weight: 700;">
                ⭐ KẾT QUẢ CUỐI CÙNG - Mô hình Stacking Ensemble ⭐
              </h4>
              <div class="pd-stacking-container">
                <div
                  class="pd-card pd-card-stacking"
                  :class="getRiskClass(predictionResult.pd_stacking)"
                >
                  <div class="pd-label-stacking">🎯 PD - Stacking</div>
                  <div class="pd-value-stacking">{{ (predictionResult.pd_stacking * 100).toFixed(2) }}%</div>
                  <div class="pd-status-stacking">{{ getRiskLabel(predictionResult.pd_stacking) }}</div>
                </div>
              </div>
            </div>

            <!-- Chart so sánh PD -->
            <div class="chart-container" style="margin-top: 2rem;">
              <RiskChart :prediction="predictionResult" />
            </div>
          </div>

          <!-- Gemini Analysis Section -->
          <div style="margin: 3rem 0;">
            <button
              @click="analyzeWithGemini"
              class="btn btn-primary"
              :disabled="isAnalyzing"
              style="width: 100%;"
            >
              {{ isAnalyzing ? '⏳ Đang phân tích...' : '🤖 Phân tích chuyên sâu bằng AI' }}
            </button>

            <div v-if="geminiAnalysis" class="analysis-box">
              <h3 style="margin-bottom: 1rem; color: #FF6B9D; font-size: 1.4rem;">
                🧠 Phân tích & Khuyến nghị từ AI
              </h3>

              <!-- Quyết định cuối cùng CHO VAY / KHÔNG CHO VAY -->
              <div class="lending-decision" :class="getLendingDecisionClass()">
                <div class="decision-icon">{{ getLendingDecisionIcon() }}</div>
                <div class="decision-text">{{ getLendingDecisionText() }}</div>
              </div>

              <div class="analysis-content">{{ geminiAnalysis }}</div>
            </div>

            <!-- Nút Phân tích sâu kết hợp Bối cảnh ngành -->
            <div style="margin-top: 2rem; text-align: center;">
              <button
                @click="goToPdIndustryTab"
                class="btn btn-accent"
                style="padding: 0.8rem 2rem; font-size: 1rem;"
              >
                🎯 Phân tích sâu kết hợp Bối cảnh ngành
              </button>
            </div>
          </div>

          <!-- Export Report Button -->
          <div v-if="geminiAnalysis" style="margin: 2rem 0; text-align: center;">
            <button
              @click="exportReport"
              class="btn btn-secondary"
              :disabled="isExporting"
              style="padding: 1rem 3rem; font-size: 1.1rem;"
            >
              {{ isExporting ? '⏳ Đang xuất báo cáo...' : '📄 Xuất Báo cáo Word' }}
            </button>
          </div>

          <!-- Chatbot Trigger - Hiện sau khi có phân tích -->
          <div v-if="geminiAnalysis && !showChatbot" class="chatbot-trigger">
            <div class="pointer-hand">👉</div>
            <div class="trigger-text" @click="openChatbot">Hỏi thêm chi tiết tại đây...</div>
          </div>
        </div>
        </div>
      </div>

      <!-- Chatbot Component -->
      <div v-if="showChatbot" class="chatbot-container">
        <div class="chatbot-header">
          <div class="chatbot-title">
            <span class="chatbot-icon">🤖</span>
            <span>Trợ lý ảo Agribank</span>
          </div>
          <button @click="closeChatbot" class="chatbot-close">✕</button>
        </div>
        <div class="chatbot-messages">
          <div v-if="chatMessages.length === 0" class="chatbot-welcome">
            <p>👋 Xin chào! Tôi là Trợ lý ảo Agribank.</p>
            <p>Bạn có thể hỏi thêm về phân tích vừa rồi.</p>
          </div>
          <div
            v-for="(message, index) in chatMessages"
            :key="index"
            class="chat-message"
            :class="{ 'user-message': message.role === 'user', 'assistant-message': message.role === 'assistant' }"
          >
            {{ message.content }}
          </div>
          <div v-if="isChatLoading" class="chat-loading">
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
          </div>
        </div>
        <div class="chatbot-input">
          <input
            v-model="chatInput"
            @keyup.enter="sendChatMessage"
            type="text"
            placeholder="Nhập câu hỏi của bạn..."
            class="chat-input-field"
          />
          <button @click="sendChatMessage" class="chat-send-button" :disabled="!chatInput.trim() || isChatLoading">
            ➤
          </button>
        </div>
      </div>

      <!-- ✅ TAB CONTENT: Dashboard Tài Chính -->
      <div v-if="activeTab === 'dashboard'" class="tab-content">
        <div class="card">
          <h2 class="card-title">📊 Dashboard Tài Chính - Phân tích Ngành nghề</h2>

          <!-- Sub-tabs cho Dashboard -->
          <div class="sub-tabs-container" style="margin: 1.5rem 0;">
            <button
              @click="dashboardSubTab = 'industry'"
              class="sub-tab-button"
              :class="{ active: dashboardSubTab === 'industry' }"
            >
              📈 Phân tích Ngành
            </button>
            <button
              @click="dashboardSubTab = 'pd-industry'"
              class="sub-tab-button"
              :class="{ active: dashboardSubTab === 'pd-industry' }"
            >
              🎯 Kết hợp Phân tích PD chuyên sâu
            </button>
          </div>

          <!-- SUB-TAB 1: Phân tích Ngành (GIỮ NGUYÊN) -->
          <div v-if="dashboardSubTab === 'industry'">
            <!-- Bảng mô tả và hướng dẫn sử dụng -->
            <div class="dashboard-guide">
              <h3 style="color: #FF6B9D; font-size: 1.1rem; margin-bottom: 0.8rem;">
                📋 Giới thiệu Dashboard
              </h3>
              <p style="margin-bottom: 0.5rem; line-height: 1.6;">
                Dashboard Tài Chính giúp bạn phân tích xu hướng và dữ liệu kinh tế theo từng ngành nghề tại Việt Nam.
                Hệ thống sử dụng AI (Gemini) để thu thập, phân tích dữ liệu mới nhất và đưa ra khuyến nghị cho quyết định tín dụng.
              </p>
              <div class="guide-steps">
                <div class="guide-step">
                  <span class="step-number">1</span>
                  <span class="step-text">Chọn ngành nghề muốn phân tích</span>
                </div>
                <div class="guide-step">
                  <span class="step-number">2</span>
                  <span class="step-text">Nhấn "🔄 AI Lấy dữ liệu" để thu thập thông tin mới nhất</span>
                </div>
                <div class="guide-step">
                  <span class="step-number">3</span>
                  <span class="step-text">Nhấn "📊 Xem biểu đồ" để hiển thị dữ liệu trực quan + phân tích sơ bộ</span>
                </div>
                <div class="guide-step">
                  <span class="step-number">4</span>
                  <span class="step-text">Nhấn "🔍 Phân tích sâu" để AI đánh giá ảnh hưởng đến quyết định cho vay</span>
                </div>
              </div>
            </div>

            <!-- Dropdown chọn ngành -->
            <div style="margin: 2rem 0;">
              <label class="input-label" style="font-size: 1rem; margin-bottom: 0.8rem;">
                🏢 Chọn ngành nghề để phân tích:
              </label>
              <select
                v-model="selectedIndustry"
                class="input-field"
                style="font-size: 1rem; padding: 0.8rem;"
              >
                <option value="">-- Chọn ngành nghề --</option>
                <option value="overview">📈 Tổng quan Kinh tế Việt Nam</option>
                <option value="agriculture">🌾 Nông nghiệp</option>
                <option value="forestry">🌲 Lâm nghiệp</option>
                <option value="fishing">🐟 Thủy sản</option>
                <option value="manufacturing">🏭 Sản xuất công nghiệp</option>
                <option value="processing">⚙️ Chế biến</option>
                <option value="construction">🏗️ Xây dựng</option>
                <option value="realestate">🏘️ Bất động sản</option>
                <option value="retail">🛒 Bán lẻ</option>
                <option value="wholesale">📦 Bán sỉ</option>
                <option value="trading">💼 Thương mại</option>
                <option value="finance">🏦 Tài chính</option>
                <option value="banking">🏧 Ngân hàng</option>
                <option value="insurance">🛡️ Bảo hiểm</option>
                <option value="technology">💻 Công nghệ Thông tin</option>
                <option value="software">📱 Phần mềm</option>
                <option value="transportation">🚚 Vận tải</option>
                <option value="logistics">📮 Logistics</option>
                <option value="tourism">✈️ Du lịch</option>
                <option value="hospitality">🏨 Khách sạn - Nhà hàng</option>
                <option value="services">🎯 Dịch vụ</option>
                <option value="healthcare">🏥 Y tế</option>
                <option value="pharmaceutical">💊 Dược phẩm</option>
                <option value="energy">⚡ Năng lượng</option>
                <option value="electricity">🔌 Điện lực</option>
                <option value="mining">⛏️ Khai khoáng</option>
                <option value="education">🎓 Giáo dục</option>
                <option value="media">📺 Truyền thông</option>
                <option value="textile">👔 Dệt may</option>
                <option value="food">🍔 Thực phẩm & Đồ uống</option>
              </select>
            </div>

            <!-- Các nút chức năng theo luồng -->
            <div v-if="selectedIndustry" class="dashboard-actions">
              <button
                @click="fetchIndustryData"
                class="btn btn-primary"
                :disabled="isFetchingData"
                style="width: 100%; margin-bottom: 1rem;"
              >
                {{ isFetchingData ? '⏳ Đang lấy dữ liệu...' : '🔄 AI Lấy dữ liệu tự động' }}
              </button>

              <button
                @click="showCharts"
                class="btn btn-secondary"
                :disabled="!industryData || isShowingCharts"
                style="width: 100%; margin-bottom: 1rem;"
              >
                {{ isShowingCharts ? '⏳ Đang tạo biểu đồ...' : '📊 Xem biểu đồ & Phân tích sơ bộ' }}
              </button>

              <button
                @click="deepAnalyze"
                class="btn btn-accent"
                :disabled="!chartsData || isDeepAnalyzing"
                style="width: 100%;"
              >
                {{ isDeepAnalyzing ? '⏳ Đang phân tích sâu...' : '🔍 Phân tích sâu - Đánh giá tín dụng' }}
              </button>
            </div>

            <!-- Kết quả: Hiển thị biểu đồ -->
            <div v-if="chartsData" class="charts-section" style="margin-top: 2rem;">
              <h3 style="color: #FF6B9D; font-size: 1.3rem; margin-bottom: 1rem; text-align: center;">
                📊 Biểu đồ dữ liệu: {{ getIndustryName(selectedIndustry) }}
              </h3>
              <div id="industry-charts-container" style="width: 100%; min-height: 400px;"></div>

              <div v-if="briefAnalysis" class="analysis-box" style="margin-top: 1.5rem;">
                <h4 style="color: #FF6B9D; font-size: 1.1rem; margin-bottom: 1rem;">
                  🤖 Phân tích sơ bộ từ AI
                </h4>
                <div class="analysis-content" style="font-size: 0.95rem; line-height: 1.7;">
                  {{ briefAnalysis }}
                </div>
              </div>
            </div>

            <!-- Kết quả: Phân tích sâu -->
            <div v-if="deepAnalysisResult" class="deep-analysis-section" style="margin-top: 2rem;">
              <div class="analysis-box" style="border: 3px solid #FF6B9D;">
                <h3 style="color: #FF1493; font-size: 1.4rem; margin-bottom: 1.5rem; text-align: center; font-weight: 900;">
                  🎯 Phân tích sâu - Đánh giá tín dụng
                </h3>
                <div class="analysis-content" style="font-size: 1rem; line-height: 1.8; font-weight: 600;">
                  {{ deepAnalysisResult }}
                </div>
              </div>

              <!-- Chatbot Trigger cho sub-tab Phân tích Ngành -->
              <div v-if="!showDashboardChatbot" class="chatbot-trigger" style="margin-top: 1.5rem;">
                <div class="pointer-hand">👉</div>
                <div class="trigger-text" @click="openDashboardChatbot">Hỏi thêm chi tiết về phân tích ngành tại đây...</div>
              </div>
            </div>
          </div>

          <!-- SUB-TAB 2: Kết hợp Phân tích PD chuyên sâu (MỚI) -->
          <div v-if="dashboardSubTab === 'pd-industry'">
            <!-- Hướng dẫn sử dụng -->
            <div class="dashboard-guide" style="margin-bottom: 2rem;">
              <h3 style="color: #9C27B0; font-size: 1.1rem; margin-bottom: 0.8rem;">
                🎯 Giới thiệu Phân tích PD kết hợp Ngành nghề
              </h3>
              <p style="margin-bottom: 0.5rem; line-height: 1.6;">
                Tính năng này cho phép phân tích chuyên sâu 14 chỉ số tài chính của doanh nghiệp kết hợp với đặc thù ngành nghề,
                giúp đưa ra khuyến nghị cho vay chính xác hơn.
              </p>
              <div class="guide-steps">
                <div class="guide-step">
                  <span class="step-number">1</span>
                  <span class="step-text">Chọn ngành nghề của doanh nghiệp</span>
                </div>
                <div class="guide-step">
                  <span class="step-number">2</span>
                  <span class="step-text">Chọn nguồn chỉ số: từ Tab Dự báo hoặc tải file mới</span>
                </div>
                <div class="guide-step">
                  <span class="step-number">3</span>
                  <span class="step-text">Nhấn "Phân tích" để xem kết quả và biểu đồ</span>
                </div>
              </div>
            </div>

            <!-- Chọn ngành nghề -->
            <div style="margin: 1.5rem 0;">
              <label class="input-label" style="font-size: 1rem; margin-bottom: 0.8rem;">
                🏢 Chọn ngành nghề của doanh nghiệp:
              </label>
              <select
                v-model="pdIndustrySelected"
                class="input-field"
                style="font-size: 1rem; padding: 0.8rem;"
              >
                <option value="">-- Chọn ngành nghề --</option>
                <option value="agriculture">🌾 Nông nghiệp</option>
                <option value="forestry">🌲 Lâm nghiệp</option>
                <option value="fishing">🐟 Thủy sản</option>
                <option value="manufacturing">🏭 Sản xuất công nghiệp</option>
                <option value="processing">⚙️ Chế biến</option>
                <option value="construction">🏗️ Xây dựng</option>
                <option value="realestate">🏘️ Bất động sản</option>
                <option value="retail">🛒 Bán lẻ</option>
                <option value="wholesale">📦 Bán sỉ</option>
                <option value="trading">💼 Thương mại</option>
                <option value="finance">🏦 Tài chính</option>
                <option value="banking">🏧 Ngân hàng</option>
                <option value="insurance">🛡️ Bảo hiểm</option>
                <option value="technology">💻 Công nghệ Thông tin</option>
                <option value="software">📱 Phần mềm</option>
                <option value="transportation">🚚 Vận tải</option>
                <option value="logistics">📮 Logistics</option>
                <option value="tourism">✈️ Du lịch</option>
                <option value="hospitality">🏨 Khách sạn - Nhà hàng</option>
                <option value="services">🎯 Dịch vụ</option>
                <option value="healthcare">🏥 Y tế</option>
                <option value="pharmaceutical">💊 Dược phẩm</option>
                <option value="energy">⚡ Năng lượng</option>
                <option value="electricity">🔌 Điện lực</option>
                <option value="mining">⛏️ Khai khoáng</option>
                <option value="education">🎓 Giáo dục</option>
                <option value="media">📺 Truyền thông</option>
                <option value="textile">👔 Dệt may</option>
                <option value="food">🍔 Thực phẩm & Đồ uống</option>
              </select>
            </div>

            <!-- Radio buttons: Chọn nguồn chỉ số -->
            <div v-if="pdIndustrySelected" style="margin: 1.5rem 0;">
              <label class="input-label" style="font-size: 1rem; margin-bottom: 0.8rem;">
                📊 Chọn nguồn chỉ số tài chính:
              </label>
              <div style="display: flex; gap: 1.5rem; margin-top: 1rem;">
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                  <input
                    type="radio"
                    v-model="pdDataSource"
                    value="from-predict"
                    style="width: 18px; height: 18px; cursor: pointer;"
                  />
                  <span style="font-size: 0.95rem; font-weight: 600;">Lấy chỉ số từ Tab Dự Báo</span>
                </label>
                <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
                  <input
                    type="radio"
                    v-model="pdDataSource"
                    value="new-file"
                    style="width: 18px; height: 18px; cursor: pointer;"
                  />
                  <span style="font-size: 0.95rem; font-weight: 600;">Tải lên File mới để phân tích</span>
                </label>
              </div>
            </div>

            <!-- Upload file mới (nếu chọn "new-file") -->
            <div v-if="pdDataSource === 'new-file'" style="margin: 1.5rem 0;">
              <div class="upload-area" @click="$refs.pdXlsxFileInput.click()" style="padding: 1rem; min-height: 80px;">
                <div class="upload-icon" style="font-size: 1.5rem;">📊</div>
                <p class="upload-text">{{ pdXlsxFileName || 'Tải lên file XLSX của doanh nghiệp' }}</p>
                <p class="upload-hint" style="font-size: 0.7rem;">
                  File XLSX phải có 3 sheets: CDKT, BCTN, LCTT
                </p>
              </div>
              <input
                ref="pdXlsxFileInput"
                type="file"
                accept=".xlsx,.xls"
                @change="handlePdXlsxFile"
                style="display: none"
              />
            </div>

            <!-- Nút phân tích -->
            <div v-if="pdDataSource" style="margin: 1.5rem 0;">
              <button
                @click="analyzePdWithIndustry"
                class="btn btn-accent"
                :disabled="isAnalyzingPdIndustry || (pdDataSource === 'from-predict' && !indicatorsDict) || (pdDataSource === 'new-file' && !pdXlsxFile)"
                style="width: 100%; padding: 1rem; font-size: 1.05rem;"
              >
                {{ isAnalyzingPdIndustry ? '⏳ Đang phân tích...' : '🎯 Phân tích PD kết hợp Ngành nghề' }}
              </button>
              <p v-if="pdDataSource === 'from-predict' && !indicatorsDict" style="color: #ff6b9d; text-align: center; margin-top: 0.5rem; font-size: 0.85rem;">
                ⚠️ Vui lòng tải file và tính toán chỉ số ở Tab "Dự Báo PD" trước
              </p>
            </div>

            <!-- Hiển thị 14 chỉ số (nhỏ gọn) -->
            <div v-if="pdAnalysisIndicators" style="margin: 2rem 0;">
              <h3 style="color: #9C27B0; font-size: 1.1rem; margin-bottom: 1rem; text-align: center;">
                📈 14 Chỉ số Tài chính đã tính toán
              </h3>
              <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.8rem;">
                <div v-for="(value, key) in pdAnalysisIndicators" :key="key"
                     style="background: linear-gradient(135deg, rgba(156, 39, 176, 0.1) 0%, rgba(233, 216, 253, 0.2) 100%);
                            padding: 0.6rem; border-radius: 8px; border: 1px solid rgba(156, 39, 176, 0.2);">
                  <div style="font-size: 0.75rem; font-weight: 700; color: #9C27B0; margin-bottom: 0.2rem;">{{ key }}</div>
                  <div style="font-size: 0.85rem; font-weight: 600; color: #4A4A4A;">{{ value.toFixed(4) }}</div>
                </div>
              </div>
            </div>

            <!-- Hiển thị biểu đồ -->
            <div v-if="pdAnalysisCharts" class="charts-section" style="margin-top: 2rem;">
              <h3 style="color: #9C27B0; font-size: 1.2rem; margin-bottom: 1rem; text-align: center;">
                📊 Biểu đồ Phân tích Chỉ số
              </h3>
              <div id="pd-industry-charts-container" style="width: 100%; min-height: 400px;"></div>
            </div>

            <!-- Hiển thị phân tích từ Gemini -->
            <div v-if="pdAnalysisResult" class="deep-analysis-section" style="margin-top: 2rem;">
              <div class="analysis-box" style="border: 3px solid #9C27B0;">
                <h3 style="color: #9C27B0; font-size: 1.3rem; margin-bottom: 1.5rem; text-align: center; font-weight: 900;">
                  🎯 Phân tích PD kết hợp Ngành nghề
                </h3>
                <div class="analysis-content" style="font-size: 0.95rem; line-height: 1.7; font-weight: 600; white-space: pre-wrap;">
                  {{ pdAnalysisResult }}
                </div>
              </div>

              <!-- Chatbot Trigger cho sub-tab PD chuyên sâu -->
              <div v-if="!showDashboardChatbot" class="chatbot-trigger" style="margin-top: 1.5rem;">
                <div class="pointer-hand">👉</div>
                <div class="trigger-text" @click="openDashboardChatbot">Hỏi thêm chi tiết về phân tích PD kết hợp ngành tại đây...</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Chatbot Component cho Dashboard -->
        <div v-if="showDashboardChatbot" class="chatbot-container">
          <div class="chatbot-header">
            <div class="chatbot-title">
              <span class="chatbot-icon">🤖</span>
              <span>Trợ lý ảo Agribank - Dashboard</span>
            </div>
            <button @click="closeDashboardChatbot" class="chatbot-close">✕</button>
          </div>
          <div class="chatbot-messages">
            <div v-if="dashboardChatMessages.length === 0" class="chatbot-welcome">
              <p>👋 Xin chào! Tôi là Trợ lý ảo Agribank.</p>
              <p>Bạn có thể hỏi thêm về phân tích Dashboard vừa rồi.</p>
            </div>
            <div
              v-for="(message, index) in dashboardChatMessages"
              :key="index"
              class="chat-message"
              :class="{ 'user-message': message.role === 'user', 'assistant-message': message.role === 'assistant' }"
            >
              {{ message.content }}
            </div>
            <div v-if="isDashboardChatLoading" class="chat-loading">
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
            </div>
          </div>
          <div class="chatbot-input">
            <input
              v-model="dashboardChatInput"
              @keyup.enter="sendDashboardChatMessage"
              type="text"
              placeholder="Nhập câu hỏi của bạn..."
              class="chat-input-field"
            />
            <button @click="sendDashboardChatMessage" class="chat-send-button" :disabled="!dashboardChatInput.trim() || isDashboardChatLoading">
              ➤
            </button>
          </div>
        </div>
      </div>

      <!-- ✅ TAB CONTENT: Mô phỏng kịch bản xấu -->
      <div v-if="activeTab === 'scenario'" class="tab-content">
        <div class="card">
          <h2 class="card-title">⚠️ Mô phỏng Kịch bản Biến động Kinh tế</h2>

          <!-- Ghi chú hướng dẫn -->
          <div class="info-note">
            <span class="note-icon">📝</span>
            <span class="note-text">Mô phỏng tác động của các kịch bản kinh tế xấu đến xác suất vỡ nợ (PD) và phân tích khả năng chịu đựng của doanh nghiệp</span>
          </div>

          <!-- Bước 1: Chọn nguồn dữ liệu -->
          <div style="margin-bottom: 2rem;">
            <h3 style="margin-bottom: 1rem; color: #FF6B9D;">📁 Bước 1: Chọn nguồn dữ liệu</h3>
            <div class="radio-group">
              <label class="radio-label">
                <input type="radio" value="from_tab" v-model="scenarioDataSource" />
                <span>Sử dụng dữ liệu từ Tab "Dự Báo PD"</span>
                <span v-if="!indicatorsDict" style="color: #999; font-size: 0.85rem; margin-left: 0.5rem;">(Chưa có dữ liệu - Vui lòng dự báo PD trước)</span>
              </label>
              <label class="radio-label">
                <input type="radio" value="new_file" v-model="scenarioDataSource" />
                <span>Tải file XLSX mới để mô phỏng</span>
              </label>
            </div>

            <!-- Upload file mới (nếu chọn new_file) -->
            <div v-if="scenarioDataSource === 'new_file'" style="margin-top: 1rem;">
              <div class="upload-area" @click="$refs.scenarioFileInput.click()">
                <div class="upload-icon">📊</div>
                <p class="upload-text">{{ scenarioFileName || 'Tải lên file XLSX của doanh nghiệp' }}</p>
                <p class="upload-hint">File XLSX phải có 3 sheets: CDKT, BCTN, LCTT</p>
              </div>
              <input
                ref="scenarioFileInput"
                type="file"
                accept=".xlsx,.xls"
                @change="handleScenarioFile"
                style="display: none"
              />
            </div>
          </div>

          <!-- Bước 2: Chọn kịch bản -->
          <div style="margin-bottom: 2rem;">
            <h3 style="margin-bottom: 1rem; color: #FF6B9D;">⚡ Bước 2: Chọn Kịch bản Biến động</h3>
            <div class="scenario-cards">
              <div
                class="scenario-card"
                :class="{ selected: selectedScenario === 'mild' }"
                @click="selectedScenario = 'mild'"
              >
                <div class="scenario-icon">🟠</div>
                <h4 class="scenario-title">Kinh tế giảm nhẹ</h4>
                <ul class="scenario-details">
                  <li>Doanh thu thuần <span class="highlight-negative">↓5%</span></li>
                  <li>Lãi suất vay <span class="highlight-negative">↑10%</span></li>
                  <li>Giá vốn hàng bán <span class="highlight-negative">↑3%</span></li>
                  <li>Thanh khoản TSNH <span class="highlight-negative">↓5%</span></li>
                </ul>
              </div>

              <div
                class="scenario-card"
                :class="{ selected: selectedScenario === 'moderate' }"
                @click="selectedScenario = 'moderate'"
              >
                <div class="scenario-icon">🔴</div>
                <h4 class="scenario-title">Cú sốc kinh tế trung bình</h4>
                <ul class="scenario-details">
                  <li>Doanh thu thuần <span class="highlight-negative">↓12%</span></li>
                  <li>Lãi suất vay <span class="highlight-negative">↑25%</span></li>
                  <li>Giá vốn hàng bán <span class="highlight-negative">↑8%</span></li>
                  <li>Thanh khoản TSNH <span class="highlight-negative">↓12%</span></li>
                </ul>
              </div>

              <div
                class="scenario-card"
                :class="{ selected: selectedScenario === 'crisis' }"
                @click="selectedScenario = 'crisis'"
              >
                <div class="scenario-icon">⚫</div>
                <h4 class="scenario-title">Khủng hoảng</h4>
                <ul class="scenario-details">
                  <li>Doanh thu thuần <span class="highlight-negative">↓25%</span></li>
                  <li>Lãi suất vay <span class="highlight-negative">↑40%</span></li>
                  <li>Giá vốn hàng bán <span class="highlight-negative">↑15%</span></li>
                  <li>Thanh khoản TSNH <span class="highlight-negative">↓25%</span></li>
                </ul>
              </div>

              <div
                class="scenario-card"
                :class="{ selected: selectedScenario === 'custom' }"
                @click="selectedScenario = 'custom'"
              >
                <div class="scenario-icon">🟡</div>
                <h4 class="scenario-title">Tùy chọn biến động</h4>
                <p class="scenario-hint">Tự điều chỉnh % biến động</p>
              </div>
            </div>

            <!-- Custom scenario inputs -->
            <div v-if="selectedScenario === 'custom'" class="custom-scenario-inputs">
              <h4 style="margin-bottom: 1rem;">Nhập tỷ lệ biến động (% âm = giảm, % dương = tăng):</h4>
              <div class="input-grid">
                <div class="input-group">
                  <label>Doanh thu thuần (%):</label>
                  <input type="number" v-model.number="customRevenue" step="0.1" placeholder="-5" />
                </div>
                <div class="input-group">
                  <label>Lãi suất vay (%):</label>
                  <input type="number" v-model.number="customInterest" step="0.1" placeholder="+10" />
                </div>
                <div class="input-group">
                  <label>Giá vốn hàng bán (%):</label>
                  <input type="number" v-model.number="customCogs" step="0.1" placeholder="+3" />
                </div>
                <div class="input-group">
                  <label>Thanh khoản TSNH (%):</label>
                  <input type="number" v-model.number="customLiquidity" step="0.1" placeholder="-5" />
                </div>
              </div>
            </div>
          </div>

          <!-- Nút bắt đầu mô phỏng -->
          <button
            @click="runScenarioSimulation"
            class="btn btn-primary"
            :disabled="!canRunSimulation || isSimulating"
            style="width: 100%; margin-bottom: 2rem;"
          >
            {{ isSimulating ? '⏳ Đang mô phỏng...' : '🎯 Bắt đầu Mô phỏng' }}
          </button>

          <!-- Kết quả mô phỏng -->
          <div v-if="scenarioResult">
            <!-- Thông tin kịch bản -->
            <div class="scenario-info-banner">
              <h3>{{ scenarioResult.scenario_info.name }}</h3>
              <div class="scenario-changes">
                <span>Doanh thu: {{ scenarioResult.scenario_info.changes.revenue >= 0 ? '+' : '' }}{{ scenarioResult.scenario_info.changes.revenue }}%</span>
                <span>Lãi suất: {{ scenarioResult.scenario_info.changes.interest >= 0 ? '+' : '' }}{{ scenarioResult.scenario_info.changes.interest }}%</span>
                <span>Giá vốn: {{ scenarioResult.scenario_info.changes.cogs >= 0 ? '+' : '' }}{{ scenarioResult.scenario_info.changes.cogs }}%</span>
                <span>Thanh khoản: {{ scenarioResult.scenario_info.changes.liquidity >= 0 ? '+' : '' }}{{ scenarioResult.scenario_info.changes.liquidity }}%</span>
              </div>
            </div>

            <!-- % Thay đổi PD - Thiết kế mới -->
            <div class="pd-change-section">
              <div class="pd-comparison-header">
                <h3 style="color: #FF6B9D; font-size: 1.5rem; margin: 0;">
                  💫 Kết quả Mô phỏng Tác động
                </h3>
              </div>

              <div class="pd-comparison-cards">
                <!-- Card Trước -->
                <div class="pd-card pd-before-card">
                  <div class="pd-card-header">
                    <span class="pd-card-icon">🟢</span>
                    <span class="pd-card-title">Trước kịch bản</span>
                  </div>
                  <div class="pd-card-value">
                    {{ (scenarioResult.pd_change.before * 100).toFixed(2) }}%
                  </div>
                  <div class="pd-card-label">Xác suất vỡ nợ (PD)</div>
                </div>

                <!-- Arrow -->
                <div class="pd-arrow-container">
                  <div class="pd-arrow">
                    <span style="font-size: 2.5rem; color: #FF6B9D;">→</span>
                  </div>
                  <div class="pd-change-badge" :class="getPdChangeClass(scenarioResult.pd_change.change_pct)">
                    <span class="change-icon">{{ scenarioResult.pd_change.change_pct >= 0 ? '⬆' : '⬇' }}</span>
                    <span class="change-value">{{ scenarioResult.pd_change.change_pct >= 0 ? '+' : '' }}{{ scenarioResult.pd_change.change_pct }}%</span>
                  </div>
                </div>

                <!-- Card Sau -->
                <div class="pd-card pd-after-card">
                  <div class="pd-card-header">
                    <span class="pd-card-icon">🔴</span>
                    <span class="pd-card-title">Sau kịch bản</span>
                  </div>
                  <div class="pd-card-value">
                    {{ (scenarioResult.pd_change.after * 100).toFixed(2) }}%
                  </div>
                  <div class="pd-card-label">Xác suất vỡ nợ (PD)</div>
                </div>
              </div>

              <!-- Nhận xét ngắn gọn -->
              <div class="pd-analysis-note">
                <div class="note-icon">💡</div>
                <div class="note-content">
                  <strong>Nhận xét:</strong>
                  <span v-if="scenarioResult.pd_change.change_pct > 50">
                    Kịch bản <strong>{{ scenarioResult.scenario_info.name }}</strong> tác động <strong style="color: #dc3545;">CỰC KỲ NGHIÊM TRỌNG</strong> đến khả năng trả nợ.
                    Xác suất vỡ nợ tăng <strong>{{ scenarioResult.pd_change.change_pct }}%</strong>, cần <strong>xem xét kỹ lưỡng</strong> trước khi cấp tín dụng.
                  </span>
                  <span v-else-if="scenarioResult.pd_change.change_pct > 20">
                    Kịch bản <strong>{{ scenarioResult.scenario_info.name }}</strong> có tác động <strong style="color: #fd7e14;">ĐÁNG KỂ</strong> đến khả năng trả nợ.
                    PD tăng <strong>{{ scenarioResult.pd_change.change_pct }}%</strong>, khuyến nghị <strong>thận trọng</strong> và có biện pháp giảm thiểu rủi ro.
                  </span>
                  <span v-else-if="scenarioResult.pd_change.change_pct > 5">
                    Kịch bản <strong>{{ scenarioResult.scenario_info.name }}</strong> tác động <strong style="color: #ffc107;">VỪA PHẢI</strong> đến rủi ro vỡ nợ.
                    PD tăng <strong>{{ scenarioResult.pd_change.change_pct }}%</strong>, doanh nghiệp vẫn <strong>chịu đựng được</strong> nhưng cần theo dõi.
                  </span>
                  <span v-else-if="scenarioResult.pd_change.change_pct > 0">
                    Kịch bản <strong>{{ scenarioResult.scenario_info.name }}</strong> có tác động <strong style="color: #28a745;">NHẸ</strong> đến khả năng trả nợ.
                    PD chỉ tăng <strong>{{ scenarioResult.pd_change.change_pct }}%</strong>, doanh nghiệp <strong>khá ổn định</strong> trong điều kiện bất lợi.
                  </span>
                  <span v-else-if="scenarioResult.pd_change.change_pct === 0">
                    Không có thay đổi đáng kể về PD. Doanh nghiệp <strong>duy trì ổn định</strong>.
                  </span>
                  <span v-else>
                    Kịch bản <strong>{{ scenarioResult.scenario_info.name }}</strong> dẫn đến <strong style="color: #28a745;">CẢI THIỆN</strong> PD (giảm {{ Math.abs(scenarioResult.pd_change.change_pct) }}%).
                    Đây là dấu hiệu <strong>tích cực</strong>.
                  </span>
                </div>
              </div>
            </div>

            <!-- 2 Bảng so sánh nằm ngang -->
            <div style="margin: 3rem 0;">
              <h3 style="margin-bottom: 1.5rem; color: #FF6B9D; text-align: center; font-size: 1.6rem;">
                📊 So sánh 14 Chỉ số Tài chính (Trước / Sau kịch bản)
              </h3>
              <div class="comparison-tables-container">
                <!-- Bảng Trước kịch bản -->
                <div class="comparison-table-wrapper">
                  <h4 class="table-subtitle">Trước kịch bản (Bình thường)</h4>
                  <table class="indicators-table">
                    <thead>
                      <tr>
                        <th>Chỉ số</th>
                        <th>Giá trị</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="indicator in scenarioResult.indicators_before" :key="indicator.code">
                        <td>
                          <div class="indicator-code-cell">{{ indicator.code }}</div>
                          <div class="indicator-name-cell">{{ indicator.name }}</div>
                        </td>
                        <td class="indicator-value-cell">{{ indicator.value.toFixed(4) }}</td>
                      </tr>
                    </tbody>
                  </table>
                  <div class="pd-summary">
                    <strong>PD (Stacking):</strong> {{ (scenarioResult.prediction_before.pd_stacking * 100).toFixed(2) }}%
                  </div>
                </div>

                <!-- Bảng Sau kịch bản -->
                <div class="comparison-table-wrapper">
                  <h4 class="table-subtitle">Sau kịch bản ({{ scenarioResult.scenario_info.name }})</h4>
                  <table class="indicators-table">
                    <thead>
                      <tr>
                        <th>Chỉ số</th>
                        <th>Giá trị</th>
                        <th>Thay đổi</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(indicator, index) in scenarioResult.indicators_after" :key="indicator.code">
                        <td>
                          <div class="indicator-code-cell">{{ indicator.code }}</div>
                          <div class="indicator-name-cell">{{ indicator.name }}</div>
                        </td>
                        <td class="indicator-value-cell">{{ indicator.value.toFixed(4) }}</td>
                        <td class="change-cell" :class="getChangeClass(indicator.value, scenarioResult.indicators_before[index].value)">
                          {{ getChangeText(indicator.value, scenarioResult.indicators_before[index].value) }}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  <div class="pd-summary">
                    <strong>PD (Stacking):</strong> {{ (scenarioResult.prediction_after.pd_stacking * 100).toFixed(2) }}%
                  </div>
                </div>
              </div>
            </div>

            <!-- 2 Biểu đồ so sánh PD (nằm ngang) -->
            <div style="margin: 3rem 0;">
              <h3 style="margin-bottom: 1.5rem; color: #FF6B9D; text-align: center; font-size: 1.6rem;">
                📊 So sánh PD Trước và Sau Biến động Kinh tế
              </h3>
              <div class="charts-comparison-container">
                <div class="chart-wrapper">
                  <h4 class="chart-title">🟢 Trước kịch bản (Bình thường)</h4>
                  <RiskChart :prediction="scenarioResult.prediction_before" />
                </div>
                <div class="chart-wrapper">
                  <h4 class="chart-title">🔴 Sau kịch bản ({{ scenarioResult.scenario_info.name }})</h4>
                  <RiskChart :prediction="scenarioResult.prediction_after" />
                </div>
              </div>
            </div>

            <!-- Nút phân tích Gemini -->
            <button
              v-if="!scenarioAnalysis"
              @click="analyzeScenario"
              class="btn btn-secondary"
              :disabled="isAnalyzingScenario"
              style="width: 100%; margin: 2rem 0;"
            >
              {{ isAnalyzingScenario ? '⏳ Đang phân tích...' : '🤖 Phân tích chuyên sâu bằng Gemini AI' }}
            </button>

            <!-- Kết quả phân tích Gemini -->
            <div v-if="scenarioAnalysis" class="gemini-analysis-section">
              <h3 style="margin-bottom: 1rem; color: #FF6B9D;">🤖 Phân tích Chuyên sâu từ Gemini AI</h3>
              <div class="analysis-content" style="white-space: pre-wrap;">{{ scenarioAnalysis }}</div>
            </div>

            <!-- Chatbot Trigger - Hiện sau khi có phân tích -->
            <div v-if="scenarioAnalysis && !showScenarioChatbot" class="chatbot-trigger">
              <div class="pointer-hand">👉</div>
              <div class="trigger-text" @click="openScenarioChatbot">Hỏi thêm chi tiết tại đây...</div>
            </div>

            <!-- Nút xuất báo cáo Word -->
            <div v-if="scenarioAnalysis" style="margin-top: 2rem; text-align: center;">
              <button @click="exportScenarioReport" class="btn btn-export" :disabled="isExportingScenario">
                {{ isExportingScenario ? '⏳ Đang xuất...' : '📄 Xuất Báo cáo Word' }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Scenario Chatbot Component - Nằm ngoài card -->
      <div v-if="showScenarioChatbot && activeTab === 'scenario'" class="chatbot-container">
        <div class="chatbot-header">
          <div class="chatbot-title">
            <span class="chatbot-icon">🤖</span>
            <span>Trợ lý ảo Agribank</span>
          </div>
          <button @click="closeScenarioChatbot" class="chatbot-close">✕</button>
        </div>
        <div class="chatbot-messages">
          <div v-if="scenarioChatMessages.length === 0" class="chatbot-welcome">
            <p>👋 Xin chào! Tôi là Trợ lý ảo Agribank.</p>
            <p>Bạn có thể hỏi thêm về phân tích mô phỏng kịch bản vừa rồi.</p>
          </div>
          <div
            v-for="(message, index) in scenarioChatMessages"
            :key="index"
            class="chat-message"
            :class="{ 'user-message': message.role === 'user', 'assistant-message': message.role === 'assistant' }"
          >
            {{ message.content }}
          </div>
          <div v-if="isScenarioChatLoading" class="chat-loading">
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
            <span class="loading-dot"></span>
          </div>
        </div>
        <div class="chatbot-input">
          <input
            v-model="scenarioChatInput"
            @keyup.enter="sendScenarioChatMessage"
            type="text"
            placeholder="Nhập câu hỏi của bạn..."
            class="chat-input-field"
          />
          <button @click="sendScenarioChatMessage" class="chat-send-button" :disabled="!scenarioChatInput.trim() || isScenarioChatLoading">
            ➤
          </button>
        </div>
      </div>

      <!-- ✅ TAB CONTENT: Mô phỏng Vĩ mô -->
      <div v-if="activeTab === 'macro'" class="tab-content">
        <div class="card">
          <h2 class="card-title">📊 Mô phỏng Vĩ mô - Stress Testing</h2>

          <!-- Ghi chú hướng dẫn -->
          <div class="info-note">
            <span class="note-icon">📝</span>
            <span class="note-text">Mô phỏng tác động của các biến vĩ mô (GDP, lạm phát, lãi suất NHNN, tỷ giá) đến khả năng trả nợ của doanh nghiệp thông qua kênh truyền dẫn Macro-to-Micro</span>
          </div>

          <!-- Bước 1: Chọn nguồn dữ liệu -->
          <div style="margin-bottom: 2rem;">
            <h3 style="margin-bottom: 1rem; color: #3B82F6;">📁 Bước 1: Chọn nguồn dữ liệu</h3>
            <div class="radio-group">
              <label class="radio-label">
                <input type="radio" value="from_tab" v-model="macroDataSource" />
                <span>Sử dụng dữ liệu từ Tab "Dự Báo PD"</span>
                <span v-if="!indicatorsDict" style="color: #999; font-size: 0.85rem; margin-left: 0.5rem;">(Chưa có dữ liệu - Vui lòng dự báo PD trước)</span>
              </label>
              <label class="radio-label">
                <input type="radio" value="new_file" v-model="macroDataSource" />
                <span>Tải file XLSX mới để mô phỏng</span>
              </label>
            </div>

            <!-- Upload file mới (nếu chọn new_file) -->
            <div v-if="macroDataSource === 'new_file'" style="margin-top: 1rem;">
              <div class="upload-area" @click="$refs.macroFileInput.click()">
                <div class="upload-icon">📊</div>
                <p class="upload-text">{{ macroFileName || 'Tải lên file XLSX của doanh nghiệp' }}</p>
                <p class="upload-hint">File XLSX phải có 3 sheets: CDKT, BCTN, LCTT</p>
              </div>
              <input
                ref="macroFileInput"
                type="file"
                accept=".xlsx,.xls"
                @change="handleMacroFile"
                style="display: none"
              />
            </div>
          </div>

          <!-- Bước 2: Chọn kịch bản vĩ mô -->
          <div style="margin-bottom: 2rem;">
            <h3 style="margin-bottom: 1rem; color: #3B82F6;">🌍 Bước 2: Chọn Kịch bản Vĩ mô</h3>
            <div class="scenario-cards">
              <div
                class="scenario-card macro-card"
                :class="{ selected: selectedMacroScenario === 'recession_mild' }"
                @click="selectedMacroScenario = 'recession_mild'"
              >
                <div class="scenario-icon">🟠</div>
                <h4 class="scenario-title">Suy thoái nhẹ</h4>
                <ul class="scenario-details">
                  <li>GDP: <span class="highlight-negative">-1.5%</span></li>
                  <li>CPI: <span class="highlight-negative">6.0%</span></li>
                  <li>PPI: <span class="highlight-negative">8.0%</span></li>
                  <li>Lãi suất NHNN: <span class="highlight-negative">+100 bps</span></li>
                  <li>Tỷ giá USD/VND: <span class="highlight-negative">+3.0%</span></li>
                </ul>
              </div>

              <div
                class="scenario-card macro-card"
                :class="{ selected: selectedMacroScenario === 'recession_moderate' }"
                @click="selectedMacroScenario = 'recession_moderate'"
              >
                <div class="scenario-icon">🔴</div>
                <h4 class="scenario-title">Suy thoái trung bình</h4>
                <ul class="scenario-details">
                  <li>GDP: <span class="highlight-negative">-3.5%</span></li>
                  <li>CPI: <span class="highlight-negative">10.0%</span></li>
                  <li>PPI: <span class="highlight-negative">14.0%</span></li>
                  <li>Lãi suất NHNN: <span class="highlight-negative">+200 bps</span></li>
                  <li>Tỷ giá USD/VND: <span class="highlight-negative">+6.0%</span></li>
                </ul>
              </div>

              <div
                class="scenario-card macro-card"
                :class="{ selected: selectedMacroScenario === 'crisis' }"
                @click="selectedMacroScenario = 'crisis'"
              >
                <div class="scenario-icon">⚫</div>
                <h4 class="scenario-title">Khủng hoảng</h4>
                <ul class="scenario-details">
                  <li>GDP: <span class="highlight-negative">-6.0%</span></li>
                  <li>CPI: <span class="highlight-negative">15.0%</span></li>
                  <li>PPI: <span class="highlight-negative">20.0%</span></li>
                  <li>Lãi suất NHNN: <span class="highlight-negative">+300 bps</span></li>
                  <li>Tỷ giá USD/VND: <span class="highlight-negative">+10.0%</span></li>
                </ul>
              </div>

              <div
                class="scenario-card macro-card"
                :class="{ selected: selectedMacroScenario === 'custom' }"
                @click="selectedMacroScenario = 'custom'"
              >
                <div class="scenario-icon">🟡</div>
                <h4 class="scenario-title">Tùy chỉnh vĩ mô</h4>
                <p class="scenario-hint">Tự điều chỉnh các biến vĩ mô</p>
              </div>
            </div>

            <!-- Custom macro scenario inputs -->
            <div v-if="selectedMacroScenario === 'custom'" class="custom-scenario-inputs">
              <h4 style="margin-bottom: 1rem;">Nhập giá trị các biến vĩ mô:</h4>
              <div class="input-grid">
                <div class="input-group">
                  <label>GDP tăng trưởng (%):</label>
                  <input type="number" v-model.number="customGdp" step="0.1" placeholder="-3.5" />
                </div>
                <div class="input-group">
                  <label>Lạm phát CPI (%):</label>
                  <input type="number" v-model.number="customCpi" step="0.1" placeholder="10.0" />
                </div>
                <div class="input-group">
                  <label>Lạm phát PPI (%):</label>
                  <input type="number" v-model.number="customPpi" step="0.1" placeholder="14.0" />
                </div>
                <div class="input-group">
                  <label>Lãi suất NHNN (bps):</label>
                  <input type="number" v-model.number="customPolicyRate" step="10" placeholder="200" />
                </div>
                <div class="input-group">
                  <label>Tỷ giá USD/VND (%):</label>
                  <input type="number" v-model.number="customFx" step="0.1" placeholder="6.0" />
                </div>
              </div>
            </div>
          </div>

          <!-- Bước 3: Chọn ngành nghề -->
          <div style="margin-bottom: 2rem;">
            <h3 style="margin-bottom: 1rem; color: #3B82F6;">🏭 Bước 3: Chọn Ngành nghề</h3>
            <select v-model="selectedIndustryCode" class="input-field" style="font-size: 1rem; padding: 0.8rem;">
              <option value="manufacturing">🏭 Sản xuất</option>
              <option value="export">📦 Xuất khẩu</option>
              <option value="retail">🛒 Bán lẻ</option>
            </select>
            <p style="margin-top: 0.5rem; color: #666; font-size: 0.9rem;">
              Ngành nghề ảnh hưởng đến hệ số nhạy cảm trong kênh truyền dẫn Macro → Micro
            </p>
          </div>

          <!-- Nút bắt đầu mô phỏng -->
          <button
            @click="runMacroSimulation"
            class="btn btn-primary"
            :disabled="!canRunMacroSimulation || isSimulatingMacro"
            style="width: 100%; margin-bottom: 2rem;"
          >
            {{ isSimulatingMacro ? '⏳ Đang mô phỏng...' : '🎯 Bắt đầu Mô phỏng Vĩ mô' }}
          </button>

          <!-- Kết quả mô phỏng vĩ mô -->
          <div v-if="macroResult">
            <!-- Banner kịch bản vĩ mô -->
            <div class="macro-scenario-banner">
              <h3>{{ macroResult.scenario_info.name }} - Ngành: {{ macroResult.scenario_info.industry }}</h3>
              <div class="macro-variables-grid">
                <span>GDP: {{ macroResult.macro_variables.gdp_growth_pct >= 0 ? '+' : '' }}{{ macroResult.macro_variables.gdp_growth_pct }}%</span>
                <span>CPI: {{ macroResult.macro_variables.inflation_cpi_pct }}%</span>
                <span>PPI: {{ macroResult.macro_variables.inflation_ppi_pct }}%</span>
                <span>Lãi suất NHNN: +{{ macroResult.macro_variables.policy_rate_change_bps }} bps</span>
                <span>Tỷ giá: +{{ macroResult.macro_variables.fx_usd_vnd_pct }}%</span>
              </div>
            </div>

            <!-- Box Chuyển đổi Macro → Micro -->
            <div class="macro-to-micro-box">
              <h3 style="color: #3B82F6; font-size: 1.4rem; margin-bottom: 1rem; text-align: center;">
                🔄 Kênh truyền dẫn: Macro → Micro
              </h3>
              <p style="text-align: center; color: #666; margin-bottom: 1.5rem;">
                Các biến vĩ mô được chuyển đổi thành biến vi mô thông qua hệ số nhạy cảm ngành
              </p>
              <div class="micro-shocks-grid">
                <div class="micro-shock-card">
                  <div class="micro-icon">💰</div>
                  <div class="micro-label">Doanh thu thuần</div>
                  <div class="micro-value" :class="{ negative: macroResult.micro_shocks.revenue_change_pct < 0 }">
                    {{ macroResult.micro_shocks.revenue_change_pct >= 0 ? '+' : '' }}{{ macroResult.micro_shocks.revenue_change_pct }}%
                  </div>
                </div>
                <div class="micro-shock-card">
                  <div class="micro-icon">📦</div>
                  <div class="micro-label">Giá vốn hàng bán</div>
                  <div class="micro-value" :class="{ negative: macroResult.micro_shocks.cogs_change_pct > 0 }">
                    {{ macroResult.micro_shocks.cogs_change_pct >= 0 ? '+' : '' }}{{ macroResult.micro_shocks.cogs_change_pct }}%
                  </div>
                </div>
                <div class="micro-shock-card">
                  <div class="micro-icon">💹</div>
                  <div class="micro-label">Lãi suất vay</div>
                  <div class="micro-value" :class="{ negative: macroResult.micro_shocks.interest_rate_change_pct > 0 }">
                    {{ macroResult.micro_shocks.interest_rate_change_pct >= 0 ? '+' : '' }}{{ macroResult.micro_shocks.interest_rate_change_pct }}%
                  </div>
                </div>
                <div class="micro-shock-card">
                  <div class="micro-icon">💧</div>
                  <div class="micro-label">Thanh khoản TSNH</div>
                  <div class="micro-value" :class="{ negative: macroResult.micro_shocks.liquidity_shock_pct < 0 }">
                    {{ macroResult.micro_shocks.liquidity_shock_pct >= 0 ? '+' : '' }}{{ macroResult.micro_shocks.liquidity_shock_pct }}%
                  </div>
                </div>
              </div>
            </div>

            <!-- So sánh PD Trước/Sau - Giống tab scenario -->
            <div class="pd-change-section">
              <div class="pd-comparison-header">
                <h3 style="color: #3B82F6; font-size: 1.5rem; margin: 0;">
                  💫 Kết quả Mô phỏng Tác động
                </h3>
              </div>

              <div class="pd-comparison-cards">
                <!-- Card Trước -->
                <div class="pd-card pd-before-card">
                  <div class="pd-card-header">
                    <span class="pd-card-icon">🟢</span>
                    <span class="pd-card-title">Trước kịch bản vĩ mô</span>
                  </div>
                  <div class="pd-card-value">
                    {{ (macroResult.pd_change.before * 100).toFixed(2) }}%
                  </div>
                  <div class="pd-card-label">Xác suất vỡ nợ (PD)</div>
                </div>

                <!-- Arrow -->
                <div class="pd-arrow-container">
                  <div class="pd-arrow">
                    <span style="font-size: 2.5rem; color: #3B82F6;">→</span>
                  </div>
                  <div class="pd-change-badge" :class="getPdChangeClass(macroResult.pd_change.change_pct)">
                    <span class="change-icon">{{ macroResult.pd_change.change_pct >= 0 ? '⬆' : '⬇' }}</span>
                    <span class="change-value">{{ macroResult.pd_change.change_pct >= 0 ? '+' : '' }}{{ macroResult.pd_change.change_pct }}%</span>
                  </div>
                </div>

                <!-- Card Sau -->
                <div class="pd-card pd-after-card">
                  <div class="pd-card-header">
                    <span class="pd-card-icon">🔴</span>
                    <span class="pd-card-title">Sau kịch bản vĩ mô</span>
                  </div>
                  <div class="pd-card-value">
                    {{ (macroResult.pd_change.after * 100).toFixed(2) }}%
                  </div>
                  <div class="pd-card-label">Xác suất vỡ nợ (PD)</div>
                </div>
              </div>

              <!-- Nhận xét ngắn gọn -->
              <div class="pd-analysis-note">
                <div class="note-icon">💡</div>
                <div class="note-content">
                  <strong>Nhận xét:</strong>
                  <span v-if="macroResult.pd_change.change_pct > 50">
                    Kịch bản vĩ mô <strong>{{ macroResult.scenario_info.name }}</strong> tác động <strong style="color: #dc3545;">CỰC KỲ NGHIÊM TRỌNG</strong> đến khả năng trả nợ.
                    Xác suất vỡ nợ tăng <strong>{{ macroResult.pd_change.change_pct }}%</strong>, cần <strong>xem xét kỹ lưỡng</strong> trước khi cấp tín dụng.
                  </span>
                  <span v-else-if="macroResult.pd_change.change_pct > 20">
                    Kịch bản vĩ mô <strong>{{ macroResult.scenario_info.name }}</strong> có tác động <strong style="color: #fd7e14;">ĐÁNG KỂ</strong> đến khả năng trả nợ.
                    PD tăng <strong>{{ macroResult.pd_change.change_pct }}%</strong>, khuyến nghị <strong>thận trọng</strong> và có biện pháp giảm thiểu rủi ro.
                  </span>
                  <span v-else-if="macroResult.pd_change.change_pct > 5">
                    Kịch bản vĩ mô <strong>{{ macroResult.scenario_info.name }}</strong> tác động <strong style="color: #ffc107;">VỪA PHẢI</strong> đến rủi ro vỡ nợ.
                    PD tăng <strong>{{ macroResult.pd_change.change_pct }}%</strong>, doanh nghiệp vẫn <strong>chịu đựng được</strong> nhưng cần theo dõi.
                  </span>
                  <span v-else-if="macroResult.pd_change.change_pct > 0">
                    Kịch bản vĩ mô <strong>{{ macroResult.scenario_info.name }}</strong> có tác động <strong style="color: #28a745;">NHẸ</strong> đến khả năng trả nợ.
                    PD chỉ tăng <strong>{{ macroResult.pd_change.change_pct }}%</strong>, doanh nghiệp <strong>khá ổn định</strong> trong điều kiện bất lợi.
                  </span>
                  <span v-else-if="macroResult.pd_change.change_pct === 0">
                    Không có thay đổi đáng kể về PD. Doanh nghiệp <strong>duy trì ổn định</strong>.
                  </span>
                  <span v-else>
                    Kịch bản vĩ mô <strong>{{ macroResult.scenario_info.name }}</strong> dẫn đến <strong style="color: #28a745;">CẢI THIỆN</strong> PD (giảm {{ Math.abs(macroResult.pd_change.change_pct) }}%).
                    Đây là dấu hiệu <strong>tích cực</strong>.
                  </span>
                </div>
              </div>
            </div>

            <!-- 2 Bảng so sánh 14 chỉ số (giống tab scenario) -->
            <div style="margin: 3rem 0;">
              <h3 style="margin-bottom: 1.5rem; color: #3B82F6; text-align: center; font-size: 1.6rem;">
                📊 So sánh 14 Chỉ số Tài chính (Trước / Sau kịch bản vĩ mô)
              </h3>
              <div class="comparison-tables-container">
                <!-- Bảng Trước kịch bản -->
                <div class="comparison-table-wrapper">
                  <h4 class="table-subtitle">Trước kịch bản (Bình thường)</h4>
                  <table class="indicators-table">
                    <thead>
                      <tr>
                        <th>Chỉ số</th>
                        <th>Giá trị</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="indicator in macroResult.indicators_before" :key="indicator.code">
                        <td>
                          <div class="indicator-code-cell">{{ indicator.code }}</div>
                          <div class="indicator-name-cell">{{ indicator.name }}</div>
                        </td>
                        <td class="indicator-value-cell">{{ indicator.value.toFixed(4) }}</td>
                      </tr>
                    </tbody>
                  </table>
                  <div class="pd-summary">
                    <strong>PD (Stacking):</strong> {{ (macroResult.prediction_before.pd_stacking * 100).toFixed(2) }}%
                  </div>
                </div>

                <!-- Bảng Sau kịch bản -->
                <div class="comparison-table-wrapper">
                  <h4 class="table-subtitle">Sau kịch bản ({{ macroResult.scenario_info.name }})</h4>
                  <table class="indicators-table">
                    <thead>
                      <tr>
                        <th>Chỉ số</th>
                        <th>Giá trị</th>
                        <th>Thay đổi</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(indicator, index) in macroResult.indicators_after" :key="indicator.code">
                        <td>
                          <div class="indicator-code-cell">{{ indicator.code }}</div>
                          <div class="indicator-name-cell">{{ indicator.name }}</div>
                        </td>
                        <td class="indicator-value-cell">{{ indicator.value.toFixed(4) }}</td>
                        <td class="change-cell" :class="getChangeClass(indicator.value, macroResult.indicators_before[index].value)">
                          {{ getChangeText(indicator.value, macroResult.indicators_before[index].value) }}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  <div class="pd-summary">
                    <strong>PD (Stacking):</strong> {{ (macroResult.prediction_after.pd_stacking * 100).toFixed(2) }}%
                  </div>
                </div>
              </div>
            </div>

            <!-- 2 Biểu đồ so sánh PD -->
            <div style="margin: 3rem 0;">
              <h3 style="margin-bottom: 1.5rem; color: #3B82F6; text-align: center; font-size: 1.6rem;">
                📊 So sánh PD Trước và Sau Kịch bản Vĩ mô
              </h3>
              <div class="charts-comparison-container">
                <div class="chart-wrapper">
                  <h4 class="chart-title">🟢 Trước kịch bản (Bình thường)</h4>
                  <RiskChart :prediction="macroResult.prediction_before" />
                </div>
                <div class="chart-wrapper">
                  <h4 class="chart-title">🔴 Sau kịch bản ({{ macroResult.scenario_info.name }})</h4>
                  <RiskChart :prediction="macroResult.prediction_after" />
                </div>
              </div>
            </div>

            <!-- Gemini Analysis Section -->
            <div style="margin: 3rem 0;">
              <button
                @click="analyzeMacro"
                class="btn btn-primary"
                :disabled="isAnalyzingMacro"
                style="width: 100%;"
              >
                {{ isAnalyzingMacro ? '⏳ Đang phân tích...' : '🤖 Phân tích sâu bằng Gemini AI' }}
              </button>

              <div v-if="macroAnalysis" class="analysis-box" style="margin-top: 2rem;">
                <h3 style="margin-bottom: 1rem; color: #FF6B9D; font-size: 1.4rem;">
                  🧠 Phân tích chuyên sâu từ AI
                </h3>
                <div class="analysis-content">{{ macroAnalysis }}</div>
              </div>
            </div>

            <!-- Chatbot Button -->
            <div v-if="macroAnalysis" style="margin-top: 2rem; text-align: center;">
              <button
                @click="openMacroChatbot"
                class="btn btn-accent"
                style="padding: 0.8rem 2rem; font-size: 1rem;"
              >
                💬 Hỏi thêm chi tiết về kết quả mô phỏng
              </button>
            </div>
          </div>
        </div>

        <!-- Chatbot Component for Macro -->
        <div v-if="showMacroChatbot" class="chatbot-container">
          <div class="chatbot-header">
            <div class="chatbot-title">
              <span class="chatbot-icon">🤖</span>
              <span>Trợ lý ảo Agribank</span>
            </div>
            <button @click="closeMacroChatbot" class="chatbot-close">✕</button>
          </div>
          <div class="chatbot-messages">
            <div v-if="macroChatMessages.length === 0" class="chatbot-welcome">
              <p>👋 Xin chào! Tôi là Trợ lý ảo Agribank.</p>
              <p>Bạn có thể hỏi thêm về kết quả mô phỏng vĩ mô vừa rồi.</p>
            </div>
            <div
              v-for="(message, index) in macroChatMessages"
              :key="index"
              class="chat-message"
              :class="{ 'user-message': message.role === 'user', 'assistant-message': message.role === 'assistant' }"
            >
              {{ message.content }}
            </div>
            <div v-if="isMacroChatLoading" class="chat-loading">
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
            </div>
          </div>
          <div class="chatbot-input">
            <input
              v-model="macroChatInput"
              @keyup.enter="sendMacroChatMessage"
              type="text"
              placeholder="Nhập câu hỏi của bạn..."
              class="chat-input-field"
            />
            <button @click="sendMacroChatMessage" class="chat-send-button" :disabled="!macroChatInput.trim() || isMacroChatLoading">
              ➤
            </button>
          </div>
        </div>
      </div>

      <!-- ✅ TAB CONTENT: Huấn luyện Mô hình -->
      <div v-if="activeTab === 'train'" class="tab-content">
        <div class="card">
          <h2 class="card-title">📚 Huấn luyện Mô hình Machine Learning</h2>

          <div style="margin-bottom: 2rem;">
            <div class="upload-area" @click="$refs.trainFileInput.click()">
              <div class="upload-icon">📤</div>
              <p class="upload-text">{{ trainFileName || 'Tải lên file CSV để huấn luyện' }}</p>
              <p class="upload-hint">File CSV cần có 14 cột (X_1 đến X_14) và cột 'default'</p>
            </div>

            <input
              ref="trainFileInput"
              type="file"
              accept=".csv"
              @change="handleTrainFile"
              style="display: none"
            />

            <button
              @click="trainModel"
              class="btn btn-primary"
              :disabled="!trainFile || isTraining"
              style="margin-top: 1rem; width: 100%;"
            >
              {{ isTraining ? '⏳ Đang huấn luyện...' : '🚀 Huấn luyện Mô hình' }}
            </button>
          </div>

          <!-- Training Results -->
          <div v-if="trainResult" style="margin-top: 2rem;">
            <h3 style="margin-bottom: 1rem; color: #FF6B9D; font-size: 1.2rem;">
              ✅ Kết quả Huấn luyện
            </h3>
            <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 240, 247, 0.95) 100%); padding: 1.5rem; border-radius: 14px; border: 2px solid rgba(255, 182, 193, 0.3);">
              <p style="margin-bottom: 0.5rem;"><strong>Số mẫu Train:</strong> {{ trainResult.train_samples }}</p>
              <p style="margin-bottom: 0.5rem;"><strong>Số mẫu Test:</strong> {{ trainResult.test_samples }}</p>
              <p style="margin-bottom: 0.5rem;"><strong>Accuracy (Test):</strong> {{ (trainResult.metrics_test.accuracy * 100).toFixed(2) }}%</p>
              <p style="margin-bottom: 0;"><strong>AUC (Test):</strong> {{ (trainResult.metrics_test.auc * 100).toFixed(2) }}%</p>
            </div>
          </div>
        </div>
      </div>

      <!-- ✅ TAB CONTENT: Cảnh báo Rủi ro Sớm (Early Warning System) -->
      <div v-if="activeTab === 'early-warning'" class="tab-content">
        <div class="card early-warning-card">
          <h2 class="card-title early-warning-title">⚠️ Hệ thống Cảnh báo Rủi ro Sớm</h2>

          <!-- Hướng dẫn sử dụng -->
          <div class="info-note" style="background: linear-gradient(135deg, #FFF5F5 0%, #FFE4E1 100%); border-left: 4px solid #FF6B6B;">
            <span class="note-icon">📋</span>
            <span class="note-text">
              Hệ thống sử dụng ML (Stacking + K-Means + Gemini AI) để chẩn đoán sức khỏe tài chính doanh nghiệp.
              <br><strong>Bước 1:</strong> Train model với file 1300 DN →
              <strong>Bước 2:</strong> Upload DN cần kiểm tra →
              <strong>Bước 3:</strong> Xem kết quả chẩn đoán chi tiết.
            </span>
          </div>

          <!-- BƯỚC 1: Upload Model Training Data -->
          <div class="early-warning-section" style="margin: 2rem 0;">
            <h3 class="section-title" style="color: #FF6B6B; font-size: 1.3rem; margin-bottom: 1rem;">
              🔄 Bước 1: Train Model với dữ liệu 1300 DN
            </h3>

            <div class="upload-area" @click="$refs.ewTrainFileInput.click()">
              <div class="upload-icon">📊</div>
              <p class="upload-text">{{ ewTrainFileName || 'Tải file Excel/CSV chứa 1300 DN' }}</p>
              <p class="upload-hint">
                File cần có 14 cột (X_1 → X_14) + cột 'label' (0=không vỡ nợ, 1=vỡ nợ)
              </p>
            </div>

            <input
              ref="ewTrainFileInput"
              type="file"
              accept=".xlsx,.xls,.csv"
              @change="handleEWTrainFile"
              style="display: none"
            />

            <button
              @click="trainEarlyWarningModel"
              class="btn btn-primary"
              :disabled="!ewTrainFile || isEWTraining"
              style="margin-top: 1rem; width: 100%;"
            >
              {{ isEWTraining ? '⏳ Đang huấn luyện mô hình...' : '🔄 Huấn luyện Mô hình Cảnh báo Sớm' }}
            </button>

            <!-- Kết quả training -->
            <div v-if="ewTrainResult" style="margin-top: 1.5rem;">
              <h4 style="color: #10B981; font-size: 1.1rem; margin-bottom: 1rem;">✅ Model đã được train thành công!</h4>
              <div class="training-result-box">
                <p><strong>📊 Số mẫu:</strong> {{ ewTrainResult.num_samples }} (Tốt: {{ ewTrainResult.num_healthy }}, Vỡ nợ: {{ ewTrainResult.num_default }})</p>

                <div style="margin-top: 1rem;">
                  <strong>🎯 Top 5 Chỉ số Quan trọng nhất:</strong>
                  <div class="feature-importance-list" style="margin-top: 0.5rem;">
                    <div
                      v-for="(value, key) in getTopFeatureImportances()"
                      :key="key"
                      class="feature-importance-item"
                      style="margin-bottom: 0.5rem;"
                    >
                      <span style="font-weight: 600;">{{ key }}:</span>
                      <div class="importance-bar" style="width: {{ value * 300 }}px; background: #FF6B9D; height: 20px; border-radius: 4px; display: inline-block; margin-left: 1rem;"></div>
                      <span style="margin-left: 0.5rem;">{{ (value * 100).toFixed(2) }}%</span>
                    </div>
                  </div>
                </div>

                <p style="margin-top: 1rem;"><strong>🔍 Phân bố theo Nhóm:</strong></p>
                <div v-if="ewTrainResult.cluster_distribution" class="cluster-distribution">
                  <span v-for="(count, cluster) in ewTrainResult.cluster_distribution" :key="cluster" style="margin-right: 1rem;">
                    {{ cluster }}: {{ count }}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <!-- BƯỚC 2: Upload DN cần kiểm tra -->
          <div v-if="ewTrainResult" class="early-warning-section" style="margin: 3rem 0;">
            <h3 class="section-title" style="color: #FF6B6B; font-size: 1.3rem; margin-bottom: 1rem;">
              🩺 Bước 2: Upload DN cần kiểm tra
            </h3>

            <!-- Sub-tabs: Upload file vs Dùng dữ liệu từ Tab Dự báo PD -->
            <div class="sub-tabs-container" style="margin: 1rem 0;">
              <button
                @click="ewCheckMode = 'upload'"
                class="sub-tab-button"
                :class="{ active: ewCheckMode === 'upload' }"
              >
                📤 Upload File Mới
              </button>
              <button
                @click="ewCheckMode = 'from-predict'"
                class="sub-tab-button"
                :class="{ active: ewCheckMode === 'from-predict' }"
                :disabled="!indicatorsDict"
              >
                🔗 Dùng dữ liệu từ Tab Dự báo PD
              </button>
            </div>

            <!-- Mode: Upload File Mới -->
            <div v-if="ewCheckMode === 'upload'">
              <div class="upload-area" @click="$refs.ewCheckFileInput.click()">
                <div class="upload-icon">📄</div>
                <p class="upload-text">{{ ewCheckFileName || 'Tải file XLSX của DN cần kiểm tra' }}</p>
                <p class="upload-hint">
                  File XLSX phải có 3 sheets: CDKT, BCTN, LCTT
                </p>
              </div>

              <input
                ref="ewCheckFileInput"
                type="file"
                accept=".xlsx,.xls"
                @change="handleEWCheckFile"
                style="display: none"
              />
            </div>

            <!-- Mode: Dùng dữ liệu từ Tab Dự báo PD -->
            <div v-if="ewCheckMode === 'from-predict' && indicatorsDict">
              <div class="success-box" style="background: #E8F5E9; border: 2px solid #4CAF50; padding: 1rem; border-radius: 8px;">
                <p style="color: #2E7D32; font-weight: 600;">✅ Sẽ sử dụng 14 chỉ số từ Tab Dự báo PD</p>
              </div>
            </div>

            <!-- Chọn kỳ báo cáo (tùy chọn) -->
            <div style="margin-top: 1.5rem;">
              <label class="input-label">📅 Kỳ báo cáo (tùy chọn - chỉ để hiển thị):</label>
              <select v-model="ewReportPeriod" class="input-field">
                <option value="">-- Không chọn --</option>
                <option value="Q1/2024">Q1/2024</option>
                <option value="Q2/2024">Q2/2024</option>
                <option value="Q3/2024">Q3/2024</option>
                <option value="Q4/2024">Q4/2024</option>
                <option value="6T1/2024">6 tháng đầu năm 2024</option>
                <option value="6T2/2024">6 tháng cuối năm 2024</option>
                <option value="2024">Năm 2024</option>
              </select>
            </div>

            <!-- Chọn ngành -->
            <div style="margin-top: 1rem;">
              <label class="input-label">🏭 Chọn ngành nghề DN:</label>
              <select v-model="ewIndustryCode" class="input-field">
                <option value="manufacturing">🏭 Sản xuất (Manufacturing)</option>
                <option value="export">📦 Xuất khẩu (Export)</option>
                <option value="retail">🛒 Bán lẻ (Retail)</option>
              </select>
            </div>

            <!-- Nút Chẩn đoán -->
            <button
              @click="checkEarlyWarning"
              class="btn btn-primary"
              :disabled="(!ewCheckFile && ewCheckMode === 'upload' && !indicatorsDict) || isEWChecking"
              style="margin-top: 1.5rem; width: 100%; font-size: 1.1rem; padding: 1rem;"
            >
              {{ isEWChecking ? '⏳ Đang chẩn đoán...' : '🩺 Chẩn đoán Rủi ro' }}
            </button>
          </div>

          <!-- BƯỚC 3: Hiển thị kết quả -->
          <div v-if="ewCheckResult" class="early-warning-results" style="margin: 3rem 0;">
            <h3 class="section-title" style="color: #FF1493; font-size: 1.5rem; margin-bottom: 2rem; text-align: center; font-weight: 900;">
              📊 Bước 3: Kết quả Chẩn đoán
            </h3>

            <!-- Kỳ báo cáo -->
            <div v-if="ewCheckResult.report_period" style="text-align: center; margin-bottom: 1.5rem;">
              <span style="font-size: 1.1rem; color: #666;">📅 Kỳ báo cáo: <strong>{{ ewCheckResult.report_period }}</strong></span>
            </div>

            <!-- 1. Health Score Gauge -->
            <div class="health-score-section" style="margin-bottom: 3rem;">
              <h4 style="color: #FF6B9D; font-size: 1.2rem; margin-bottom: 1rem; text-align: center;">💚 Điểm Sức khỏe Tài chính</h4>
              <div id="health-score-gauge" style="width: 100%; height: 300px;"></div>

              <!-- Risk Level Badge -->
              <div class="risk-level-badge" :style="{ backgroundColor: ewCheckResult.risk_level_color }">
                {{ ewCheckResult.risk_level_icon }} {{ ewCheckResult.risk_level_text }}
              </div>

              <!-- Current PD -->
              <div style="text-align: center; margin-top: 1rem; font-size: 1.1rem;">
                <strong>PD hiện tại:</strong> <span :style="{ color: ewCheckResult.risk_level_color, fontSize: '1.3rem', fontWeight: 'bold' }">{{ ewCheckResult.current_pd.toFixed(2) }}%</span>
              </div>
            </div>

            <!-- 2. Top 3 Điểm Yếu -->
            <div class="weaknesses-section" style="margin-bottom: 3rem;">
              <h4 style="color: #FF6B9D; font-size: 1.2rem; margin-bottom: 1rem;">⚠️ Top 3 Điểm Yếu Cần Cải Thiện</h4>
              <div class="weakness-cards">
                <div
                  v-for="(weakness, index) in ewCheckResult.top_weaknesses"
                  :key="index"
                  class="weakness-card"
                  :class="'severity-' + weakness.severity"
                >
                  <div class="weakness-header">
                    <span class="weakness-number">#{{ index + 1 }}</span>
                    <span class="weakness-name">{{ weakness.name }}</span>
                  </div>
                  <div class="weakness-body">
                    <div class="weakness-values">
                      <div class="weakness-value">
                        <span class="value-label">Giá trị hiện tại:</span>
                        <span class="value-number">{{ weakness.current_value.toFixed(4) }}</span>
                      </div>
                      <div class="weakness-value">
                        <span class="value-label">Ngưỡng an toàn:</span>
                        <span class="value-number">{{ weakness.safe_threshold.toFixed(4) }}</span>
                      </div>
                      <div class="weakness-value">
                        <span class="value-label">Khoảng cách (Gap):</span>
                        <span class="value-number" :style="{ color: weakness.gap < 0 ? '#EF4444' : '#10B981' }">
                          {{ weakness.gap.toFixed(4) }}
                        </span>
                      </div>
                      <div class="weakness-value">
                        <span class="value-label">Percentile:</span>
                        <span class="value-number">{{ weakness.percentile.toFixed(1) }}%</span>
                      </div>
                    </div>
                    <!-- Mini bar chart cho gap -->
                    <div class="weakness-gap-chart">
                      <div class="gap-bar-container">
                        <div
                          class="gap-bar"
                          :style="{
                            width: Math.min(Math.abs(weakness.gap / weakness.safe_threshold) * 100, 100) + '%',
                            backgroundColor: weakness.gap < 0 ? '#EF4444' : '#10B981'
                          }"
                        ></div>
                      </div>
                      <div class="gap-severity-label">{{ getSeverityLabel(weakness.severity) }}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- 3. Cluster Position -->
            <div class="cluster-section" style="margin-bottom: 3rem;">
              <h4 style="color: #FF6B9D; font-size: 1.2rem; margin-bottom: 1rem;">📍 Vị trí trong 1300 DN</h4>
              <div class="cluster-info-box">
                <p style="font-size: 1.2rem; text-align: center; margin-bottom: 1rem;">
                  Bạn thuộc <strong>{{ ewCheckResult.cluster_info.cluster_name }}</strong>
                </p>
                <p style="font-size: 1rem; text-align: center; margin-bottom: 1.5rem;">
                  Xếp hạng <strong style="color: #FF6B9D; font-size: 1.3rem;">{{ ewCheckResult.cluster_info.position_percentile.toFixed(1) }}%</strong> trong 1300 DN
                </p>
                <p style="text-align: center; color: #666;">
                  PD trung bình của cluster: {{ ewCheckResult.cluster_info.cluster_avg_pd.toFixed(2) }}%
                </p>
              </div>

              <!-- Radar Chart: So sánh với median của cluster -->
              <div id="cluster-radar-chart" style="width: 100%; height: 500px; margin-top: 1.5rem;"></div>
            </div>

            <!-- 4. PD Projection Timeline -->
            <div class="pd-projection-section" style="margin-bottom: 3rem;">
              <h4 style="color: #FF6B9D; font-size: 1.2rem; margin-bottom: 1rem;">📈 Dự báo PD Tương lai (3/6/12 tháng)</h4>
              <div id="pd-projection-chart" style="width: 100%; height: 400px;"></div>
            </div>

            <!-- 5. Gemini AI Diagnosis -->
            <div class="gemini-diagnosis-section" style="margin-bottom: 2rem;">
              <h4 style="color: #FF1493; font-size: 1.3rem; margin-bottom: 1rem; text-align: center; font-weight: 900;">
                🤖 Báo cáo Chẩn đoán từ Gemini AI
              </h4>
              <div class="gemini-diagnosis-box">
                <div class="diagnosis-content" v-html="renderMarkdown(ewCheckResult.gemini_diagnosis)"></div>
              </div>
            </div>

            <!-- Chatbot Button -->
            <div style="margin-top: 2rem; text-align: center;">
              <button
                @click="openEWChatbot"
                class="btn btn-accent"
                style="padding: 0.8rem 2rem; font-size: 1rem;"
              >
                💬 Hỏi thêm chi tiết về kết quả chẩn đoán
              </button>
            </div>
          </div>
        </div>

        <!-- Chatbot Component for Early Warning -->
        <div v-if="showEWChatbot" class="chatbot-container">
          <div class="chatbot-header">
            <div class="chatbot-title">
              <span class="chatbot-icon">🤖</span>
              <span>Trợ lý ảo Agribank</span>
            </div>
            <button @click="closeEWChatbot" class="chatbot-close">✕</button>
          </div>
          <div class="chatbot-messages">
            <div v-if="ewChatMessages.length === 0" class="chatbot-welcome">
              <p>👋 Xin chào! Tôi là Trợ lý ảo Agribank.</p>
              <p>Bạn có thể hỏi thêm về kết quả chẩn đoán vừa rồi.</p>
            </div>
            <div
              v-for="(message, index) in ewChatMessages"
              :key="index"
              class="chat-message"
              :class="{ 'user-message': message.role === 'user', 'assistant-message': message.role === 'assistant' }"
            >
              {{ message.content }}
            </div>
            <div v-if="isEWChatLoading" class="chat-loading">
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
              <span class="loading-dot"></span>
            </div>
          </div>
          <div class="chatbot-input">
            <input
              v-model="ewChatInput"
              @keyup.enter="sendEWChatMessage"
              type="text"
              placeholder="Nhập câu hỏi của bạn..."
              class="chat-input-field"
            />
            <button @click="sendEWChatMessage" class="chat-send-button" :disabled="!ewChatInput.trim() || isEWChatLoading">
              ➤
            </button>
          </div>
        </div>
      </div>

      <!-- ✅ TAB CONTENT: Phát hiện Gian lận (Anomaly Detection) -->
      <div v-if="activeTab === 'anomaly'" class="tab-content">
        <div class="card anomaly-card">
          <h2 class="card-title" style="color: #FF4444; font-size: 1.8rem; text-align: center;">🚨 Hệ thống Phát hiện Bất thường</h2>

          <!-- Hướng dẫn sử dụng -->
          <div class="info-note" style="background: linear-gradient(135deg, #FFF5F5 0%, #FFE4E1 100%); border-left: 4px solid #FF4444;">
            <span class="note-icon">📋</span>
            <span class="note-text">
              <strong>Mục đích:</strong> Phát hiện doanh nghiệp có hành vi tài chính bất thường, nghi ngờ gian lận hoặc báo cáo sai lệch bằng Isolation Forest và Gemini AI.
              <br><strong>Cách sử dụng:</strong>
              <strong>Bước 1:</strong> Train model với file 1300 DN (có cột label: 0=khỏe mạnh, 1=vỡ nợ) →
              <strong>Bước 2:</strong> Upload DN cần kiểm tra hoặc dùng dữ liệu từ Tab Dự báo PD →
              <strong>Bước 3:</strong> Xem kết quả phân tích bất thường chi tiết.
            </span>
          </div>

          <!-- BƯỚC 1: Upload Model Training Data -->
          <div class="anomaly-section" style="margin: 2rem 0;">
            <h3 class="section-title" style="color: #FF4444; font-size: 1.3rem; margin-bottom: 1rem;">
              🔄 Bước 1: Train Model Phát hiện Bất thường
            </h3>

            <div class="upload-area" @click="$refs.anomalyTrainFileInput.click()" style="cursor: pointer;">
              <div class="upload-icon">📊</div>
              <p class="upload-text">{{ anomalyTrainFileName || 'Tải lên file dữ liệu 1300 DN (CSV/Excel)' }}</p>
              <p class="upload-hint">
                File phải có 14 chỉ số (X_1 → X_14) + cột 'label' (0=khỏe mạnh, 1=vỡ nợ)
              </p>
            </div>
            <input
              ref="anomalyTrainFileInput"
              type="file"
              accept=".xlsx,.xls,.csv"
              @change="handleAnomalyTrainFile"
              style="display: none"
            />

            <button
              @click="trainAnomalyModel"
              class="btn btn-primary"
              :disabled="!anomalyTrainFile || isAnomalyTraining"
              style="margin-top: 1rem; width: 100%;"
            >
              {{ isAnomalyTraining ? '⏳ Đang train model...' : '🚀 Train Model Phát hiện Bất thường' }}
            </button>

            <!-- Training Results -->
            <div v-if="anomalyTrainResult" style="margin-top: 1.5rem;">
              <h4 style="color: #10B981; font-size: 1.1rem; margin-bottom: 1rem;">✅ Model đã train thành công!</h4>

              <!-- Feature Statistics Table -->
              <div style="overflow-x: auto; margin-top: 1rem;">
                <h5 style="color: #FF4444; margin-bottom: 0.5rem;">📊 Ngưỡng an toàn của 14 chỉ số (từ DN khỏe mạnh):</h5>
                <table class="indicators-table" style="font-size: 0.85rem;">
                  <thead>
                    <tr>
                      <th>Chỉ số</th>
                      <th>P5</th>
                      <th>P50 (Trung vị)</th>
                      <th>P95</th>
                      <th>Mean</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="stat in anomalyTrainResult.feature_statistics" :key="stat.feature">
                      <td>
                        <div style="font-weight: 600;">{{ stat.feature }}</div>
                        <div style="font-size: 0.8rem; color: #666;">{{ stat.name }}</div>
                      </td>
                      <td>{{ stat.P5 }}</td>
                      <td style="font-weight: 600;">{{ stat.P50 }}</td>
                      <td>{{ stat.P95 }}</td>
                      <td>{{ stat.mean }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <p style="margin-top: 1rem; color: #666;">
                <strong>Contamination Rate:</strong> {{ (anomalyTrainResult.contamination_rate * 100).toFixed(1) }}%
                (Model giả định {{ (anomalyTrainResult.contamination_rate * 100).toFixed(1) }}% DN là bất thường)
              </p>
            </div>
          </div>

          <!-- BƯỚC 2: Upload DN cần kiểm tra -->
          <div v-if="anomalyTrainResult" class="anomaly-section" style="margin: 3rem 0; border-top: 2px solid #FFE4E1; padding-top: 2rem;">
            <h3 class="section-title" style="color: #FF4444; font-size: 1.3rem; margin-bottom: 1rem;">
              🔍 Bước 2: Upload DN cần kiểm tra Bất thường
            </h3>

            <!-- Chọn nguồn dữ liệu -->
            <div style="margin-bottom: 1.5rem;">
              <label style="display: block; margin-bottom: 0.5rem; font-weight: 600; color: #555;">
                Chọn nguồn dữ liệu:
              </label>
              <div style="display: flex; gap: 1rem;">
                <label style="display: flex; align-items: center; cursor: pointer;">
                  <input type="radio" value="from_tab" v-model="anomalyDataSource" style="margin-right: 0.5rem;" />
                  <span>Dùng dữ liệu từ Tab Dự báo PD</span>
                </label>
                <label style="display: flex; align-items: center; cursor: pointer;">
                  <input type="radio" value="upload_file" v-model="anomalyDataSource" style="margin-right: 0.5rem;" />
                  <span>Tải file mới</span>
                </label>
              </div>
            </div>

            <!-- Nếu chọn upload file -->
            <div v-if="anomalyDataSource === 'upload_file'" style="margin-bottom: 1rem;">
              <div class="upload-area" @click="$refs.anomalyCheckFileInput.click()" style="cursor: pointer;">
                <div class="upload-icon">📄</div>
                <p class="upload-text">{{ anomalyCheckFileName || 'Tải lên file XLSX của DN' }}</p>
                <p class="upload-hint">File XLSX có 3 sheets: CDKT, BCTN, LCTT</p>
              </div>
              <input
                ref="anomalyCheckFileInput"
                type="file"
                accept=".xlsx,.xls"
                @change="handleAnomalyCheckFile"
                style="display: none"
              />
            </div>

            <!-- Nếu chọn dùng dữ liệu từ Tab Dự báo PD -->
            <div v-if="anomalyDataSource === 'from_tab'" style="margin-bottom: 1rem;">
              <div v-if="!indicatorsDict" class="info-note" style="background: #FFF9E6; border-left: 4px solid #FFC107;">
                <span class="note-icon">⚠️</span>
                <span class="note-text">
                  Chưa có dữ liệu từ Tab Dự báo PD. Vui lòng vào Tab "🔮 Dự Báo PD" để tải file và tính toán 14 chỉ số trước.
                </span>
              </div>
              <div v-else class="info-note" style="background: #E8F5E9; border-left: 4px solid #10B981;">
                <span class="note-icon">✅</span>
                <span class="note-text">
                  Đã tải được 14 chỉ số từ Tab Dự báo PD. Nhấn "Kiểm tra Bất thường" để phân tích.
                </span>
              </div>
            </div>

            <button
              @click="checkAnomaly"
              class="btn btn-primary"
              :disabled="!canCheckAnomaly || isAnomalyChecking"
              style="width: 100%;"
            >
              {{ isAnomalyChecking ? '⏳ Đang kiểm tra...' : '🔍 Kiểm tra Bất thường' }}
            </button>
          </div>

          <!-- BƯỚC 3: Kết quả -->
          <div v-if="anomalyCheckResult" class="anomaly-section" style="margin: 3rem 0; border-top: 2px solid #FFE4E1; padding-top: 2rem;">
            <h3 class="section-title" style="color: #FF4444; font-size: 1.3rem; margin-bottom: 1.5rem; text-align: center;">
              📊 Bước 3: Kết quả Phân tích Bất thường
            </h3>

            <!-- Anomaly Score Gauge -->
            <div style="margin-bottom: 2rem;">
              <h4 style="color: #FF4444; font-size: 1.1rem; margin-bottom: 1rem; text-align: center;">
                🎯 Điểm Bất thường (Anomaly Score)
              </h4>
              <div id="anomaly-score-gauge" class="anomaly-score-gauge" style="width: 100%; height: 300px;"></div>
            </div>

            <!-- Risk Level Badge -->
            <div style="margin: 2rem 0; text-align: center;">
              <div class="risk-level-badge" :style="{
                background: anomalyCheckResult.risk_level_color,
                color: 'white',
                padding: '1.5rem 3rem',
                borderRadius: '16px',
                fontSize: '1.5rem',
                fontWeight: '700',
                display: 'inline-block',
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
              }">
                {{ anomalyCheckResult.risk_level_icon }} {{ anomalyCheckResult.risk_level }}
              </div>
            </div>

            <!-- Abnormal Features Table -->
            <div v-if="anomalyCheckResult.abnormal_features.length > 0" style="margin: 2rem 0;">
              <h4 style="color: #FF4444; font-size: 1.1rem; margin-bottom: 1rem;">
                ⚠️ Các chỉ số Bất thường ({{ anomalyCheckResult.abnormal_features.length }} chỉ số)
              </h4>
              <div style="overflow-x: auto;">
                <table class="abnormal-features-table">
                  <thead>
                    <tr>
                      <th>Chỉ số</th>
                      <th>Giá trị hiện tại</th>
                      <th>P5 (Ngưỡng thấp)</th>
                      <th>P50 (Trung vị)</th>
                      <th>P95 (Ngưỡng cao)</th>
                      <th>Độ lệch (%)</th>
                      <th>Mức độ</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="ab in anomalyCheckResult.abnormal_features"
                      :key="ab.feature_code"
                      :class="{ 'severity-high': ab.severity === 'high', 'severity-medium': ab.severity === 'medium' }"
                    >
                      <td>
                        <div style="font-weight: 600;">{{ ab.feature_code }}</div>
                        <div style="font-size: 0.8rem; color: #666;">{{ ab.feature_name }}</div>
                      </td>
                      <td style="font-weight: 600; color: #FF4444;">{{ ab.current_value }}</td>
                      <td>{{ ab.p5 }}</td>
                      <td>{{ ab.p50 }}</td>
                      <td>{{ ab.p95 }}</td>
                      <td style="font-weight: 600;">
                        <span v-if="ab.direction === 'low'" style="color: #EF4444;">↓ {{ ab.deviation_percent }}%</span>
                        <span v-else style="color: #F59E0B;">↑ {{ ab.deviation_percent }}%</span>
                      </td>
                      <td>
                        <span v-if="ab.severity === 'high'" style="color: #EF4444; font-weight: 600;">🔴 Cao</span>
                        <span v-else style="color: #F59E0B; font-weight: 600;">🔶 Trung bình</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div v-else style="margin: 2rem 0;">
              <div class="info-note" style="background: #E8F5E9; border-left: 4px solid #10B981;">
                <span class="note-icon">✅</span>
                <span class="note-text">
                  Không phát hiện chỉ số bất thường. Tất cả các chỉ số nằm trong ngưỡng an toàn (P5 - P95).
                </span>
              </div>
            </div>

            <!-- Comparison Radar Chart -->
            <div style="margin: 2rem 0;">
              <h4 style="color: #FF4444; font-size: 1.1rem; margin-bottom: 1rem; text-align: center;">
                📈 So sánh với DN Khỏe mạnh
              </h4>
              <div id="comparison-radar-chart" class="comparison-radar-chart" style="width: 100%; height: 500px;"></div>
            </div>

            <!-- Anomaly Type Badge -->
            <div style="margin: 2rem 0; text-align: center;">
              <h4 style="color: #FF4444; font-size: 1.1rem; margin-bottom: 0.5rem;">Loại Bất thường:</h4>
              <div class="anomaly-type-badge" style="
                display: inline-block;
                background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
                color: white;
                padding: 1rem 2rem;
                borderRadius: '12px';
                fontSize: '1.2rem';
                fontWeight: '600';
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
              ">
                {{ anomalyCheckResult.anomaly_type }}
              </div>
              <p style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
                <span v-if="anomalyCheckResult.anomaly_type === 'Normal'">✅ Doanh nghiệp hoạt động bình thường</span>
                <span v-else-if="anomalyCheckResult.anomaly_type === 'Point Anomaly'">⚠️ Bất thường tại 1 điểm riêng lẻ</span>
                <span v-else-if="anomalyCheckResult.anomaly_type === 'Contextual Anomaly'">🔶 Bất thường theo ngữ cảnh (2-4 chỉ số)</span>
                <span v-else-if="anomalyCheckResult.anomaly_type === 'Collective Anomaly'">🔴 Bất thường tập thể (≥5 chỉ số) - Nguy hiểm!</span>
              </p>
            </div>

            <!-- Gemini Explanation Box -->
            <div style="margin: 2rem 0;">
              <div class="gemini-explanation-box" style="
                background: linear-gradient(135deg, #FFF5F5 0%, #FFE4E1 100%);
                border: 3px solid #FFB6C1;
                borderRadius: '16px';
                padding: '2rem';
                boxShadow: '0 4px 12px rgba(255, 182, 193, 0.3)'
              ">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                  <span style="font-size: 2rem; margin-right: 0.5rem;">🤖</span>
                  <h4 style="color: #FF4444; font-size: 1.2rem; margin: 0;">Phân tích từ Gemini AI</h4>
                </div>
                <div style="line-height: 1.8; color: #333; white-space: pre-wrap;">{{ anomalyCheckResult.gemini_explanation }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, nextTick } from 'vue'
import axios from 'axios'
import * as echarts from 'echarts'
import RiskChart from './components/RiskChart.vue'
import IndicatorsChart from './components/IndicatorsChart.vue'

export default {
  name: 'App',
  components: {
    RiskChart,
    IndicatorsChart
  },
  setup() {
    // ✅ TAB STATE - Mặc định là 'predict'
    const activeTab = ref('predict')

    // Scroll to top button
    const showScrollTop = ref(false)
    const scrollTopPosition = ref(100)

    // Chatbot - Tab Dự báo PD
    const showChatbot = ref(false)
    const chatMessages = ref([])
    const chatInput = ref('')
    const isChatLoading = ref(false)

    // Chatbot - Dashboard Tài chính
    const showDashboardChatbot = ref(false)
    const dashboardChatMessages = ref([])
    const dashboardChatInput = ref('')
    const isDashboardChatLoading = ref(false)

    // Training
    const trainFile = ref(null)
    const trainFileName = ref('')
    const isTraining = ref(false)
    const trainResult = ref(null)

    // Prediction
    const xlsxFile = ref(null)
    const xlsxFileName = ref('')
    const isPredicting = ref(false)
    const indicators = ref([])
    const indicatorsDict = ref(null)
    const predictionResult = ref(null)

    // Gemini Analysis
    const isAnalyzing = ref(false)
    const geminiAnalysis = ref('')

    // Export
    const isExporting = ref(false)

    // Dashboard Industry Analysis - OLD (giữ lại cho tương thích)
    const selectedIndustry = ref('')
    const isAnalyzingIndustry = ref(false)
    const industryAnalysis = ref('')
    const industryCharts = ref([])

    // Dashboard Industry Analysis - NEW
    const isFetchingData = ref(false)
    const industryData = ref(null)
    const isShowingCharts = ref(false)
    const chartsData = ref(null)
    const briefAnalysis = ref('')
    const isDeepAnalyzing = ref(false)
    const deepAnalysisResult = ref('')

    // Dashboard Sub-tab State
    const dashboardSubTab = ref('industry')

    // PD + Industry Analysis - NEW FEATURE
    const pdIndustrySelected = ref('')
    const pdDataSource = ref('')
    const pdXlsxFile = ref(null)
    const pdXlsxFileName = ref('')
    const isAnalyzingPdIndustry = ref(false)
    const pdAnalysisIndicators = ref(null)
    const pdAnalysisCharts = ref(null)
    const pdAnalysisResult = ref('')

    // Scenario Simulation - NEW FEATURE
    const scenarioDataSource = ref('from_tab')
    const scenarioFile = ref(null)
    const scenarioFileName = ref('')
    const selectedScenario = ref('mild')
    const customRevenue = ref(-5)
    const customInterest = ref(10)
    const customCogs = ref(3)
    const customLiquidity = ref(-5)
    const isSimulating = ref(false)
    const scenarioResult = ref(null)
    const isAnalyzingScenario = ref(false)
    const scenarioAnalysis = ref('')
    const showScenarioChatbot = ref(false)
    const scenarioChatMessages = ref([])
    const scenarioChatInput = ref('')
    const isScenarioChatLoading = ref(false)
    const isExportingScenario = ref(false)

    // Macro Scenario Simulation - NEW FEATURE
    const macroDataSource = ref('from_tab')
    const macroFile = ref(null)
    const macroFileName = ref('')
    const selectedMacroScenario = ref('recession_mild')
    const selectedIndustryCode = ref('manufacturing')
    const customGdp = ref(-3.5)
    const customCpi = ref(10.0)
    const customPpi = ref(14.0)
    const customPolicyRate = ref(200)
    const customFx = ref(6.0)
    const isSimulatingMacro = ref(false)
    const macroResult = ref(null)
    const isAnalyzingMacro = ref(false)
    const macroAnalysis = ref('')

    // Chatbot - Macro Tab
    const showMacroChatbot = ref(false)
    const macroChatMessages = ref([])
    const macroChatInput = ref('')
    const isMacroChatLoading = ref(false)

    // Early Warning System - NEW FEATURE
    const ewTrainFile = ref(null)
    const ewTrainFileName = ref('')
    const isEWTraining = ref(false)
    const ewTrainResult = ref(null)
    const ewCheckMode = ref('upload')
    const ewCheckFile = ref(null)
    const ewCheckFileName = ref('')
    const ewReportPeriod = ref('')
    const ewIndustryCode = ref('manufacturing')
    const isEWChecking = ref(false)
    const ewCheckResult = ref(null)

    // Chatbot - Early Warning Tab
    const showEWChatbot = ref(false)
    const ewChatMessages = ref([])
    const ewChatInput = ref('')
    const isEWChatLoading = ref(false)

    // Anomaly Detection System - NEW FEATURE
    const anomalyTrainFile = ref(null)
    const anomalyTrainFileName = ref('')
    const isAnomalyTraining = ref(false)
    const anomalyTrainResult = ref(null)
    const anomalyDataSource = ref('from_tab')
    const anomalyCheckFile = ref(null)
    const anomalyCheckFileName = ref('')
    const isAnomalyChecking = ref(false)
    const anomalyCheckResult = ref(null)

    // Computed: can check anomaly
    const canCheckAnomaly = computed(() => {
      if (anomalyDataSource.value === 'from_tab') {
        return indicatorsDict.value !== null
      } else {
        return anomalyCheckFile.value !== null
      }
    })

    // API Base URL
    const API_BASE = 'http://localhost:8000'

    // Methods
    const handleTrainFile = (event) => {
      const file = event.target.files[0]
      if (file) {
        trainFile.value = file
        trainFileName.value = file.name
      }
    }

    const trainModel = async () => {
      if (!trainFile.value) return

      isTraining.value = true
      trainResult.value = null

      try {
        const formData = new FormData()
        formData.append('file', trainFile.value)

        const response = await axios.post(`${API_BASE}/train`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        trainResult.value = response.data
        alert('✅ Huấn luyện mô hình thành công!')
      } catch (error) {
        alert('❌ Lỗi khi huấn luyện: ' + (error.response?.data?.detail || error.message))
      } finally {
        isTraining.value = false
      }
    }

    const handleXlsxFile = (event) => {
      const file = event.target.files[0]
      if (file) {
        xlsxFile.value = file
        xlsxFileName.value = file.name
      }
    }

    const predictFromXlsx = async () => {
      if (!xlsxFile.value) return

      isPredicting.value = true
      indicators.value = []
      indicatorsDict.value = null
      predictionResult.value = null
      geminiAnalysis.value = ''

      try {
        const formData = new FormData()
        formData.append('file', xlsxFile.value)

        const response = await axios.post(`${API_BASE}/predict-from-xlsx`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        if (response.data.status === 'success') {
          indicators.value = response.data.indicators
          indicatorsDict.value = response.data.indicators_dict
          predictionResult.value = response.data.prediction

          alert('✅ Tính toán 14 chỉ số và dự báo PD thành công!')
        }
      } catch (error) {
        alert('❌ Lỗi khi xử lý file XLSX: ' + (error.response?.data?.detail || error.message))
      } finally {
        isPredicting.value = false
      }
    }

    const analyzeWithGemini = async () => {
      if (!predictionResult.value || !indicatorsDict.value) return

      isAnalyzing.value = true
      geminiAnalysis.value = ''

      try {
        const requestData = {
          prediction: predictionResult.value,
          indicators_dict: indicatorsDict.value,
          indicators: indicators.value
        }

        const response = await axios.post(`${API_BASE}/analyze`, requestData)

        if (response.data.status === 'success') {
          geminiAnalysis.value = response.data.analysis
        }
      } catch (error) {
        alert('❌ Lỗi khi phân tích bằng Gemini: ' + (error.response?.data?.detail || error.message))
      } finally {
        isAnalyzing.value = false
      }
    }

    const exportReport = async () => {
      if (!predictionResult.value || !geminiAnalysis.value) return

      isExporting.value = true

      try {
        const reportData = {
          prediction: predictionResult.value,
          indicators: indicators.value,
          indicators_dict: indicatorsDict.value,
          analysis: geminiAnalysis.value
        }

        const response = await axios.post(`${API_BASE}/export-report`, reportData, {
          responseType: 'blob'
        })

        // Tạo URL để download
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', `bao_cao_tin_dung_${new Date().getTime()}.docx`)
        document.body.appendChild(link)
        link.click()
        link.remove()

        alert('✅ Xuất báo cáo thành công!')
      } catch (error) {
        alert('❌ Lỗi khi xuất báo cáo: ' + (error.response?.data?.detail || error.message))
      } finally {
        isExporting.value = false
      }
    }

    const getRiskClass = (pd) => {
      const pdPercent = pd * 100
      if (pdPercent < 2) return 'risk-very-low'
      if (pdPercent < 5) return 'risk-low'
      if (pdPercent < 10) return 'risk-medium'
      if (pdPercent < 20) return 'risk-high'
      return 'risk-very-high'
    }

    const getRiskLabel = (pd) => {
      const pdPercent = pd * 100
      if (pdPercent < 2) return '🟢 Rất thấp (AAA-AA) - Doanh nghiệp xuất sắc'
      if (pdPercent < 5) return '🟢 Thấp (A-BBB) - Doanh nghiệp tốt'
      if (pdPercent < 10) return '🟡 Trung bình (BB) - Cần theo dõi'
      if (pdPercent < 20) return '🟠 Cao (B) - Rủi ro đáng kể'
      return '🔴 Rất cao (CCC-D) - Nguy cơ vỡ nợ cao'
    }

    const getLendingDecisionClass = () => {
      if (!predictionResult.value) return ''
      const pdPercent = predictionResult.value.pd_stacking * 100
      return pdPercent < 10 ? 'decision-approve' : 'decision-reject'
    }

    const getLendingDecisionIcon = () => {
      if (!predictionResult.value) return ''
      const pdPercent = predictionResult.value.pd_stacking * 100
      return pdPercent < 10 ? '✅' : '❌'
    }

    const getLendingDecisionText = () => {
      if (!predictionResult.value) return ''
      const pdPercent = predictionResult.value.pd_stacking * 100
      return pdPercent < 10 ? 'CHO VAY' : 'KHÔNG CHO VAY'
    }

    // Dashboard Industry Analysis
    const getIndustryName = (industry) => {
      const names = {
        'overview': 'Tổng quan Kinh tế Việt Nam',
        'agriculture': 'Nông nghiệp',
        'forestry': 'Lâm nghiệp',
        'fishing': 'Thủy sản',
        'manufacturing': 'Sản xuất công nghiệp',
        'processing': 'Chế biến',
        'construction': 'Xây dựng',
        'realestate': 'Bất động sản',
        'retail': 'Bán lẻ',
        'wholesale': 'Bán sỉ',
        'trading': 'Thương mại',
        'finance': 'Tài chính',
        'banking': 'Ngân hàng',
        'insurance': 'Bảo hiểm',
        'technology': 'Công nghệ Thông tin',
        'software': 'Phần mềm',
        'transportation': 'Vận tải',
        'logistics': 'Logistics',
        'tourism': 'Du lịch',
        'hospitality': 'Khách sạn - Nhà hàng',
        'services': 'Dịch vụ',
        'healthcare': 'Y tế',
        'pharmaceutical': 'Dược phẩm',
        'energy': 'Năng lượng',
        'electricity': 'Điện lực',
        'mining': 'Khai khoáng',
        'education': 'Giáo dục',
        'media': 'Truyền thông',
        'textile': 'Dệt may',
        'food': 'Thực phẩm & Đồ uống'
      }
      return names[industry] || industry
    }

    const analyzeIndustry = async () => {
      if (!selectedIndustry.value) return

      isAnalyzingIndustry.value = true
      industryAnalysis.value = ''
      industryCharts.value = []

      try {
        const requestData = {
          industry: selectedIndustry.value,
          industry_name: getIndustryName(selectedIndustry.value)
        }

        const response = await axios.post(`${API_BASE}/analyze-industry`, requestData)

        if (response.data.status === 'success') {
          industryAnalysis.value = response.data.analysis
          industryCharts.value = response.data.charts || []

          // Scroll to results
          setTimeout(() => {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
          }, 100)
        }
      } catch (error) {
        alert('❌ Lỗi khi phân tích ngành: ' + (error.response?.data?.detail || error.message))
      } finally {
        isAnalyzingIndustry.value = false
      }
    }

    // NEW Dashboard Methods
    const fetchIndustryData = async () => {
      if (!selectedIndustry.value) return

      isFetchingData.value = true
      industryData.value = null
      chartsData.value = null
      briefAnalysis.value = ''
      deepAnalysisResult.value = ''

      try {
        const requestData = {
          industry: selectedIndustry.value,
          industry_name: getIndustryName(selectedIndustry.value)
        }

        const response = await axios.post(`${API_BASE}/fetch-industry-data`, requestData)

        if (response.data.status === 'success') {
          industryData.value = response.data.data
          alert('✅ Đã lấy dữ liệu thành công! Nhấn "Xem biểu đồ" để tiếp tục.')
        }
      } catch (error) {
        alert('❌ Lỗi khi lấy dữ liệu: ' + (error.response?.data?.detail || error.message))
      } finally {
        isFetchingData.value = false
      }
    }

    const showCharts = async () => {
      if (!industryData.value) return

      isShowingCharts.value = true
      chartsData.value = null
      briefAnalysis.value = ''

      try {
        const requestData = {
          industry: selectedIndustry.value,
          industry_name: getIndustryName(selectedIndustry.value),
          data: industryData.value
        }

        const response = await axios.post(`${API_BASE}/generate-charts`, requestData)

        if (response.data.status === 'success') {
          chartsData.value = response.data.charts_data
          briefAnalysis.value = response.data.brief_analysis

          // Render charts using ECharts
          await nextTick()
          renderCharts(response.data.charts_data)
        }
      } catch (error) {
        alert('❌ Lỗi khi tạo biểu đồ: ' + (error.response?.data?.detail || error.message))
      } finally {
        isShowingCharts.value = false
      }
    }

    const renderCharts = (chartsDataArray) => {
      const container = document.getElementById('industry-charts-container')
      if (!container) return

      // Clear container
      container.innerHTML = ''

      // Tạo nhiều biểu đồ ECharts
      chartsDataArray.forEach((chartConfig, index) => {
        const chartDiv = document.createElement('div')
        chartDiv.id = `chart-${index}`
        chartDiv.style.width = '100%'
        chartDiv.style.height = '400px'
        chartDiv.style.marginBottom = '2rem'
        container.appendChild(chartDiv)

        const chartInstance = echarts.init(chartDiv)
        chartInstance.setOption(chartConfig)
      })
    }

    const deepAnalyze = async () => {
      if (!chartsData.value) return

      isDeepAnalyzing.value = true
      deepAnalysisResult.value = ''

      try {
        const requestData = {
          industry: selectedIndustry.value,
          industry_name: getIndustryName(selectedIndustry.value),
          data: industryData.value,
          brief_analysis: briefAnalysis.value
        }

        const response = await axios.post(`${API_BASE}/deep-analyze-industry`, requestData)

        if (response.data.status === 'success') {
          deepAnalysisResult.value = response.data.deep_analysis
        }
      } catch (error) {
        alert('❌ Lỗi khi phân tích sâu: ' + (error.response?.data?.detail || error.message))
      } finally {
        isDeepAnalyzing.value = false
      }
    }

    // NEW: Handle PD XLSX file upload
    const handlePdXlsxFile = (event) => {
      const file = event.target.files[0]
      if (file) {
        pdXlsxFile.value = file
        pdXlsxFileName.value = file.name
      }
    }

    // NEW: Analyze PD with Industry
    const analyzePdWithIndustry = async () => {
      if (!pdIndustrySelected.value || !pdDataSource.value) return

      isAnalyzingPdIndustry.value = true
      pdAnalysisIndicators.value = null
      pdAnalysisCharts.value = null
      pdAnalysisResult.value = ''

      try {
        let indicatorsToUse = null

        // Option 1: Lấy từ Tab Dự báo
        if (pdDataSource.value === 'from-predict') {
          if (!indicatorsDict.value) {
            alert('⚠️ Vui lòng tải file và tính toán chỉ số ở Tab "Dự Báo PD" trước')
            return
          }
          indicatorsToUse = indicatorsDict.value
        }
        // Option 2: Tải file mới
        else if (pdDataSource.value === 'new-file') {
          if (!pdXlsxFile.value) {
            alert('⚠️ Vui lòng tải lên file XLSX')
            return
          }

          // Tính toán 14 chỉ số từ file mới
          const formData = new FormData()
          formData.append('file', pdXlsxFile.value)

          const calcResponse = await axios.post(`${API_BASE}/predict-from-xlsx`, formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          })

          if (calcResponse.data.status === 'success') {
            indicatorsToUse = calcResponse.data.indicators_dict
          } else {
            alert('❌ Lỗi khi tính toán chỉ số từ file XLSX')
            return
          }
        }

        // Gọi API phân tích PD kết hợp ngành
        const requestData = {
          indicators_dict: indicatorsToUse,
          industry: pdIndustrySelected.value,
          industry_name: getIndustryName(pdIndustrySelected.value)
        }

        const response = await axios.post(`${API_BASE}/analyze-pd-with-industry`, requestData)

        if (response.data.status === 'success') {
          pdAnalysisIndicators.value = indicatorsToUse
          pdAnalysisResult.value = response.data.analysis
          pdAnalysisCharts.value = response.data.charts_data

          // Render charts
          await nextTick()
          renderPdIndustryCharts(response.data.charts_data)

          alert('✅ Phân tích PD kết hợp ngành nghề thành công!')
        }
      } catch (error) {
        alert('❌ Lỗi khi phân tích: ' + (error.response?.data?.detail || error.message))
      } finally {
        isAnalyzingPdIndustry.value = false
      }
    }

    // NEW: Render PD Industry Charts
    const renderPdIndustryCharts = (chartsDataArray) => {
      const container = document.getElementById('pd-industry-charts-container')
      if (!container) return

      // Clear container
      container.innerHTML = ''

      // Tạo nhiều biểu đồ ECharts
      chartsDataArray.forEach((chartConfig, index) => {
        const chartDiv = document.createElement('div')
        chartDiv.id = `pd-chart-${index}`
        chartDiv.style.width = '100%'
        chartDiv.style.height = '400px'
        chartDiv.style.marginBottom = '2rem'
        container.appendChild(chartDiv)

        const chartInstance = echarts.init(chartDiv)
        chartInstance.setOption(chartConfig)
      })
    }

    // Scroll to top functionality
    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop
      showScrollTop.value = scrollTop > 300

      // Cập nhật vị trí nút theo chuột
      scrollTopPosition.value = Math.min(100 + scrollTop * 0.05, window.innerHeight - 100)
    }

    const scrollToTop = () => {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      })
    }

    // Navigate to PD Industry Tab
    const goToPdIndustryTab = () => {
      activeTab.value = 'dashboard'
      dashboardSubTab.value = 'pd-industry'

      // Scroll to top
      setTimeout(() => {
        window.scrollTo({ top: 0, behavior: 'smooth' })
      }, 100)
    }

    // Chatbot functionality - Tab Dự báo PD
    const openChatbot = () => {
      showChatbot.value = true
    }

    const closeChatbot = () => {
      showChatbot.value = false
    }

    const sendChatMessage = async () => {
      if (!chatInput.value.trim() || isChatLoading.value) return

      const userMessage = chatInput.value
      chatMessages.value.push({
        role: 'user',
        content: userMessage
      })
      chatInput.value = ''
      isChatLoading.value = true

      try {
        const requestData = {
          question: userMessage,
          context: geminiAnalysis.value,
          indicators: indicatorsDict.value,
          prediction: predictionResult.value
        }

        const response = await axios.post(`${API_BASE}/chat-assistant`, requestData)

        if (response.data.status === 'success') {
          chatMessages.value.push({
            role: 'assistant',
            content: response.data.answer
          })
        }
      } catch (error) {
        chatMessages.value.push({
          role: 'assistant',
          content: '❌ Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.'
        })
      } finally {
        isChatLoading.value = false
      }
    }

    // Chatbot functionality - Dashboard Tài chính
    const openDashboardChatbot = () => {
      showDashboardChatbot.value = true
    }

    const closeDashboardChatbot = () => {
      showDashboardChatbot.value = false
    }

    const sendDashboardChatMessage = async () => {
      if (!dashboardChatInput.value.trim() || isDashboardChatLoading.value) return

      const userMessage = dashboardChatInput.value
      dashboardChatMessages.value.push({
        role: 'user',
        content: userMessage
      })
      dashboardChatInput.value = ''
      isDashboardChatLoading.value = true

      try {
        // Xác định context dựa trên sub-tab hiện tại
        let context = ''
        let indicators = {}
        let prediction = {}

        if (dashboardSubTab.value === 'industry') {
          // Sub-tab Phân tích Ngành
          context = deepAnalysisResult.value || briefAnalysis.value || 'Chưa có phân tích ngành'
          indicators = { industry: selectedIndustry.value, industry_name: getIndustryName(selectedIndustry.value) }
        } else if (dashboardSubTab.value === 'pd-industry') {
          // Sub-tab PD chuyên sâu
          context = pdAnalysisResult.value || 'Chưa có phân tích PD kết hợp ngành'
          indicators = pdAnalysisIndicators.value || {}
          prediction = { industry: pdIndustrySelected.value, industry_name: getIndustryName(pdIndustrySelected.value) }
        }

        const requestData = {
          question: userMessage,
          context: context,
          indicators: indicators,
          prediction: prediction
        }

        const response = await axios.post(`${API_BASE}/chat-assistant`, requestData)

        if (response.data.status === 'success') {
          dashboardChatMessages.value.push({
            role: 'assistant',
            content: response.data.answer
          })
        }
      } catch (error) {
        dashboardChatMessages.value.push({
          role: 'assistant',
          content: '❌ Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.'
        })
      } finally {
        isDashboardChatLoading.value = false
      }
    }

    // Chatbot functionality - Early Warning Tab
    const openEWChatbot = () => {
      showEWChatbot.value = true
    }

    const closeEWChatbot = () => {
      showEWChatbot.value = false
    }

    const sendEWChatMessage = async () => {
      if (!ewChatInput.value.trim() || isEWChatLoading.value) return

      const userMessage = ewChatInput.value
      ewChatMessages.value.push({
        role: 'user',
        content: userMessage
      })
      ewChatInput.value = ''
      isEWChatLoading.value = true

      try {
        const requestData = {
          question: userMessage,
          context: ewCheckResult.value?.gemini_diagnosis || 'Chưa có kết quả chẩn đoán',
          indicators: indicatorsDict.value || {},
          prediction: {
            health_score: ewCheckResult.value?.health_score,
            risk_level: ewCheckResult.value?.risk_level_text,
            current_pd: ewCheckResult.value?.current_pd
          }
        }

        const response = await axios.post(`${API_BASE}/chat-assistant`, requestData)

        if (response.data.status === 'success') {
          ewChatMessages.value.push({
            role: 'assistant',
            content: response.data.answer
          })
        }
      } catch (error) {
        ewChatMessages.value.push({
          role: 'assistant',
          content: '❌ Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.'
        })
      } finally {
        isEWChatLoading.value = false
      }
    }

    // ========================================================================================
    // ANOMALY DETECTION SYSTEM - METHODS
    // ========================================================================================

    const handleAnomalyTrainFile = (event) => {
      const file = event.target.files[0]
      if (file) {
        anomalyTrainFile.value = file
        anomalyTrainFileName.value = file.name
      }
    }

    const trainAnomalyModel = async () => {
      if (!anomalyTrainFile.value) return

      isAnomalyTraining.value = true
      anomalyTrainResult.value = null

      try {
        const formData = new FormData()
        formData.append('file', anomalyTrainFile.value)

        const response = await axios.post(`${API_BASE}/train-anomaly-model`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })

        if (response.data.status === 'success') {
          anomalyTrainResult.value = response.data
          alert('✅ Train Anomaly Detection Model thành công!')
        }
      } catch (error) {
        console.error('Lỗi khi train anomaly model:', error)
        alert('❌ Lỗi khi train model: ' + (error.response?.data?.detail || error.message))
      } finally {
        isAnomalyTraining.value = false
      }
    }

    const handleAnomalyCheckFile = (event) => {
      const file = event.target.files[0]
      if (file) {
        anomalyCheckFile.value = file
        anomalyCheckFileName.value = file.name
      }
    }

    const checkAnomaly = async () => {
      if (!canCheckAnomaly.value) return

      isAnomalyChecking.value = true
      anomalyCheckResult.value = null

      try {
        const formData = new FormData()

        if (anomalyDataSource.value === 'upload_file') {
          // Upload file mới
          formData.append('file', anomalyCheckFile.value)
        } else {
          // Dùng dữ liệu từ Tab Dự báo PD
          formData.append('indicators_json', JSON.stringify(indicatorsDict.value))
        }

        const response = await axios.post(`${API_BASE}/check-anomaly`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })

        if (response.data.status === 'success') {
          anomalyCheckResult.value = response.data

          // Đợi DOM cập nhật rồi render charts
          await nextTick()
          renderAnomalyScoreGauge()
          renderComparisonRadarChart()
        }
      } catch (error) {
        console.error('Lỗi khi kiểm tra bất thường:', error)
        alert('❌ Lỗi khi kiểm tra bất thường: ' + (error.response?.data?.detail || error.message))
      } finally {
        isAnomalyChecking.value = false
      }
    }

    const renderAnomalyScoreGauge = () => {
      if (!anomalyCheckResult.value) return

      const chartDom = document.getElementById('anomaly-score-gauge')
      if (!chartDom) return

      const myChart = echarts.init(chartDom)
      const score = anomalyCheckResult.value.anomaly_score

      const option = {
        series: [
          {
            type: 'gauge',
            startAngle: 180,
            endAngle: 0,
            min: 0,
            max: 100,
            splitNumber: 10,
            axisLine: {
              lineStyle: {
                width: 20,
                color: [
                  [0.6, '#10B981'],
                  [0.8, '#F59E0B'],
                  [1, '#EF4444']
                ]
              }
            },
            pointer: {
              icon: 'path://M2090.36389,615.30999 L2090.36389,615.30999 C2091.48372,615.30999 2092.40383,616.194028 2092.44859,617.312956 L2096.90698,728.755929 C2097.05155,732.369577 2094.2393,735.416212 2090.62566,735.56078 C2090.53845,735.564269 2090.45117,735.566014 2090.36389,735.566014 L2090.36389,735.566014 C2086.74736,735.566014 2083.81557,732.63423 2083.81557,729.017692 C2083.81557,728.930412 2083.81732,728.84314 2083.82081,728.755929 L2088.2792,617.312956 C2088.32396,616.194028 2089.24407,615.30999 2090.36389,615.30999 Z',
              length: '75%',
              width: 16,
              offsetCenter: [0, '5%']
            },
            axisTick: {
              length: 12,
              lineStyle: {
                color: 'auto',
                width: 2
              }
            },
            splitLine: {
              length: 20,
              lineStyle: {
                color: 'auto',
                width: 3
              }
            },
            axisLabel: {
              color: '#464646',
              fontSize: 14,
              distance: -50,
              formatter: function (value) {
                return value.toFixed(0)
              }
            },
            title: {
              offsetCenter: [0, '30%'],
              fontSize: 16,
              color: '#FF4444'
            },
            detail: {
              fontSize: 32,
              offsetCenter: [0, '60%'],
              valueAnimation: true,
              formatter: function (value) {
                return value.toFixed(1)
              },
              color: 'auto'
            },
            data: [
              {
                value: score,
                name: 'Anomaly Score'
              }
            ]
          }
        ]
      }

      myChart.setOption(option)
    }

    const renderComparisonRadarChart = () => {
      if (!anomalyCheckResult.value) return

      const chartDom = document.getElementById('comparison-radar-chart')
      if (!chartDom) return

      const myChart = echarts.init(chartDom)

      const comparison = anomalyCheckResult.value.comparison_with_healthy

      // Tạo indicator data
      const indicators = comparison.map(item => ({
        name: item.feature,
        max: Math.max(Math.abs(item.current), Math.abs(item.healthy_mean)) * 1.5 || 1
      }))

      // Tạo data series
      const currentValues = comparison.map(item => item.current)
      const healthyValues = comparison.map(item => item.healthy_mean)

      const option = {
        title: {
          text: ''
        },
        legend: {
          data: ['DN hiện tại', 'DN khỏe mạnh (Mean)'],
          top: 20
        },
        radar: {
          indicator: indicators,
          shape: 'polygon',
          splitNumber: 4
        },
        series: [
          {
            name: 'So sánh DN',
            type: 'radar',
            data: [
              {
                value: currentValues,
                name: 'DN hiện tại',
                areaStyle: {
                  color: 'rgba(255, 68, 68, 0.3)'
                },
                lineStyle: {
                  color: '#FF4444',
                  width: 2
                },
                itemStyle: {
                  color: '#FF4444'
                }
              },
              {
                value: healthyValues,
                name: 'DN khỏe mạnh (Mean)',
                areaStyle: {
                  color: 'rgba(16, 185, 129, 0.3)'
                },
                lineStyle: {
                  color: '#10B981',
                  width: 2
                },
                itemStyle: {
                  color: '#10B981'
                }
              }
            ]
          }
        ]
      }

      myChart.setOption(option)
    }

    // Chatbot functionality - Macro Tab
    const openMacroChatbot = () => {
      showMacroChatbot.value = true
    }

    const closeMacroChatbot = () => {
      showMacroChatbot.value = false
    }

    const sendMacroChatMessage = async () => {
      if (!macroChatInput.value.trim() || isMacroChatLoading.value) return

      const userMessage = macroChatInput.value
      macroChatMessages.value.push({
        role: 'user',
        content: userMessage
      })
      macroChatInput.value = ''
      isMacroChatLoading.value = true

      try {
        const requestData = {
          question: userMessage,
          context: macroAnalysis.value || 'Chưa có phân tích vĩ mô',
          indicators: macroResult.value?.indicators_after || {},
          prediction: macroResult.value?.prediction_after || {}
        }

        const response = await axios.post(`${API_BASE}/chat-assistant`, requestData)

        if (response.data.status === 'success') {
          macroChatMessages.value.push({
            role: 'assistant',
            content: response.data.answer
          })
        }
      } catch (error) {
        macroChatMessages.value.push({
          role: 'assistant',
          content: '❌ Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.'
        })
      } finally {
        isMacroChatLoading.value = false
      }
    }

    // Gemini Analysis for Macro Tab
    const analyzeMacro = async () => {
      if (!macroResult.value) return

      isAnalyzingMacro.value = true
      macroAnalysis.value = ''

      try {
        const requestData = {
          indicators_before: macroResult.value.indicators_before,
          indicators_after: macroResult.value.indicators_after,
          prediction_before: macroResult.value.prediction_before,
          prediction_after: macroResult.value.prediction_after,
          scenario_info: macroResult.value.scenario_info,
          pd_change: macroResult.value.pd_change
        }

        const response = await axios.post(`${API_BASE}/analyze-macro`, requestData)

        if (response.data.status === 'success') {
          macroAnalysis.value = response.data.analysis
        }
      } catch (error) {
        alert('❌ Lỗi khi phân tích: ' + (error.response?.data?.detail || error.message))
      } finally {
        isAnalyzingMacro.value = false
      }
    }

    // ================================================================================================
    // SCENARIO SIMULATION METHODS
    // ================================================================================================

    const handleScenarioFile = (event) => {
      const file = event.target.files[0]
      if (file) {
        scenarioFile.value = file
        scenarioFileName.value = file.name
      }
    }

    const canRunSimulation = computed(() => {
      if (scenarioDataSource.value === 'from_tab') {
        return indicatorsDict.value !== null
      } else {
        return scenarioFile.value !== null
      }
    })

    const runScenarioSimulation = async () => {
      if (!canRunSimulation.value) return

      isSimulating.value = true
      scenarioResult.value = null
      scenarioAnalysis.value = ''
      showScenarioChatbot.value = false
      scenarioChatMessages.value = []

      try {
        const formData = new FormData()

        // Thêm dữ liệu tùy theo nguồn
        if (scenarioDataSource.value === 'new_file') {
          formData.append('file', scenarioFile.value)
        } else {
          // Sử dụng dữ liệu từ Tab Dự báo PD
          formData.append('indicators_json', JSON.stringify(indicatorsDict.value))
        }

        // Thêm thông tin kịch bản
        formData.append('scenario_type', selectedScenario.value)

        if (selectedScenario.value === 'custom') {
          formData.append('custom_revenue', customRevenue.value.toString())
          formData.append('custom_interest', customInterest.value.toString())
          formData.append('custom_cogs', customCogs.value.toString())
          formData.append('custom_liquidity', customLiquidity.value.toString())
        }

        const response = await axios.post(`${API_BASE}/simulate-scenario`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        scenarioResult.value = response.data
        console.log('✅ Mô phỏng kịch bản thành công:', response.data)
      } catch (error) {
        console.error('❌ Lỗi khi mô phỏng kịch bản:', error)
        alert(error.response?.data?.detail || 'Lỗi khi mô phỏng kịch bản. Vui lòng thử lại.')
      } finally {
        isSimulating.value = false
      }
    }

    const analyzeScenario = async () => {
      if (!scenarioResult.value) return

      isAnalyzingScenario.value = true

      try {
        const response = await axios.post(`${API_BASE}/analyze-scenario`, scenarioResult.value)
        scenarioAnalysis.value = response.data.analysis
        console.log('✅ Phân tích kịch bản thành công')
      } catch (error) {
        console.error('❌ Lỗi khi phân tích kịch bản:', error)
        alert('Lỗi khi phân tích. Vui lòng kiểm tra GEMINI_API_KEY và thử lại.')
      } finally {
        isAnalyzingScenario.value = false
      }
    }

    // Scenario Chatbot functionality
    const openScenarioChatbot = () => {
      showScenarioChatbot.value = true
    }

    const closeScenarioChatbot = () => {
      showScenarioChatbot.value = false
    }

    const sendScenarioChatMessage = async () => {
      if (!scenarioChatInput.value.trim() || isScenarioChatLoading.value) return

      const userMessage = scenarioChatInput.value.trim()
      scenarioChatMessages.value.push({
        role: 'user',
        content: userMessage
      })
      scenarioChatInput.value = ''
      isScenarioChatLoading.value = true

      try {
        const response = await axios.post(`${API_BASE}/chat-assistant`, {
          question: userMessage,
          context: scenarioAnalysis.value,
          indicators: scenarioResult.value.indicators_after_dict,
          prediction: scenarioResult.value.prediction_after
        })

        if (response.data.status === 'success') {
          scenarioChatMessages.value.push({
            role: 'assistant',
            content: response.data.answer
          })
        }
      } catch (error) {
        scenarioChatMessages.value.push({
          role: 'assistant',
          content: '❌ Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.'
        })
      } finally {
        isScenarioChatLoading.value = false
      }
    }

    const exportScenarioReport = async () => {
      if (!scenarioResult.value || !scenarioAnalysis.value) return

      isExportingScenario.value = true

      try {
        // Tạo dữ liệu báo cáo
        const reportData = {
          prediction: scenarioResult.value.prediction_after,
          indicators: scenarioResult.value.indicators_after,
          indicators_dict: scenarioResult.value.indicators_after_dict,
          analysis: scenarioAnalysis.value,
          scenario_info: scenarioResult.value.scenario_info,
          comparison: {
            before: scenarioResult.value.prediction_before,
            after: scenarioResult.value.prediction_after,
            pd_change: scenarioResult.value.pd_change
          }
        }

        const response = await axios.post(`${API_BASE}/export-report`, reportData, {
          responseType: 'blob'
        })

        // Tạo link download
        const blob = new Blob([response.data], {
          type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        })
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `bao_cao_mo_phong_${Date.now()}.docx`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)

        console.log('✅ Xuất báo cáo thành công')
      } catch (error) {
        console.error('❌ Lỗi khi xuất báo cáo:', error)
        alert('Lỗi khi xuất báo cáo. Vui lòng thử lại.')
      } finally {
        isExportingScenario.value = false
      }
    }

    // ================================================================================
    // MACRO SCENARIO METHODS
    // ================================================================================
    const handleMacroFile = (event) => {
      const file = event.target.files[0]
      if (file) {
        macroFile.value = file
        macroFileName.value = file.name
      }
    }

    const canRunMacroSimulation = computed(() => {
      if (macroDataSource.value === 'from_tab') {
        return !!indicatorsDict.value
      } else if (macroDataSource.value === 'new_file') {
        return !!macroFile.value
      }
      return false
    })

    const runMacroSimulation = async () => {
      if (!canRunMacroSimulation.value) return

      isSimulatingMacro.value = true
      macroResult.value = null

      try {
        const formData = new FormData()

        // Thêm nguồn dữ liệu
        if (macroDataSource.value === 'from_tab') {
          formData.append('indicators_json', JSON.stringify(indicatorsDict.value))
        } else if (macroDataSource.value === 'new_file') {
          formData.append('file', macroFile.value)
        }

        // Thêm kịch bản vĩ mô
        formData.append('scenario_type', selectedMacroScenario.value)
        formData.append('industry_code', selectedIndustryCode.value)

        // Nếu là custom, thêm các giá trị tùy chỉnh
        if (selectedMacroScenario.value === 'custom') {
          formData.append('custom_gdp', customGdp.value)
          formData.append('custom_cpi', customCpi.value)
          formData.append('custom_ppi', customPpi.value)
          formData.append('custom_policy_rate', customPolicyRate.value)
          formData.append('custom_fx', customFx.value)
        } else {
          formData.append('custom_gdp', 0)
          formData.append('custom_cpi', 0)
          formData.append('custom_ppi', 0)
          formData.append('custom_policy_rate', 0)
          formData.append('custom_fx', 0)
        }

        const response = await axios.post(`${API_BASE}/simulate-scenario-macro`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        if (response.data.status === 'success') {
          macroResult.value = response.data
          console.log('✅ Mô phỏng vĩ mô thành công:', macroResult.value)
          alert('✅ Mô phỏng kịch bản vĩ mô thành công!')
        }
      } catch (error) {
        console.error('❌ Lỗi khi mô phỏng vĩ mô:', error)
        alert('❌ Lỗi khi mô phỏng: ' + (error.response?.data?.detail || error.message))
      } finally {
        isSimulatingMacro.value = false
      }
    }

    const getPdChangeClass = (changePct) => {
      const absChange = Math.abs(changePct)
      if (absChange < 10) return 'pd-change-low'
      if (absChange < 30) return 'pd-change-moderate'
      if (absChange < 50) return 'pd-change-high'
      return 'pd-change-critical'
    }

    const getChangeClass = (after, before) => {
      if (before === 0) return ''
      const change = ((after - before) / before) * 100
      if (Math.abs(change) < 1) return 'change-neutral'
      return change > 0 ? 'change-up' : 'change-down'
    }

    const getChangeText = (after, before) => {
      if (before === 0) return 'N/A'
      const change = ((after - before) / before) * 100
      const arrow = change > 0 ? '↑' : change < 0 ? '↓' : '→'
      return `${arrow}${Math.abs(change).toFixed(1)}%`
    }

    // ====================================================================================================
    // EARLY WARNING SYSTEM METHODS
    // ====================================================================================================

    const handleEWTrainFile = (event) => {
      const file = event.target.files[0]
      if (file) {
        ewTrainFile.value = file
        ewTrainFileName.value = file.name
      }
    }

    const trainEarlyWarningModel = async () => {
      if (!ewTrainFile.value) return

      isEWTraining.value = true
      ewTrainResult.value = null

      try {
        const formData = new FormData()
        formData.append('file', ewTrainFile.value)

        const response = await axios.post(`${API_BASE}/train-early-warning-model`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        if (response.data.status === 'success') {
          ewTrainResult.value = response.data
          alert('✅ Early Warning System trained successfully!')
        }
      } catch (error) {
        alert('❌ Lỗi khi train model: ' + (error.response?.data?.detail || error.message))
      } finally {
        isEWTraining.value = false
      }
    }

    const handleEWCheckFile = (event) => {
      const file = event.target.files[0]
      if (file) {
        ewCheckFile.value = file
        ewCheckFileName.value = file.name
      }
    }

    const checkEarlyWarning = async () => {
      if (ewCheckMode.value === 'upload' && !ewCheckFile.value) {
        alert('⚠️ Vui lòng upload file DN cần kiểm tra!')
        return
      }

      if (ewCheckMode.value === 'from-predict' && !indicatorsDict.value) {
        alert('⚠️ Chưa có dữ liệu từ Tab Dự báo PD. Vui lòng chạy dự báo PD trước!')
        return
      }

      isEWChecking.value = true
      ewCheckResult.value = null

      try {
        const formData = new FormData()

        if (ewCheckMode.value === 'upload') {
          formData.append('file', ewCheckFile.value)
        } else {
          formData.append('indicators_json', JSON.stringify(indicatorsDict.value))
        }

        if (ewReportPeriod.value) {
          formData.append('report_period', ewReportPeriod.value)
        }

        formData.append('industry_code', ewIndustryCode.value)

        const response = await axios.post(`${API_BASE}/early-warning-check`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        if (response.data.status === 'success') {
          ewCheckResult.value = response.data

          // Vẽ các biểu đồ sau khi có kết quả
          await nextTick()
          renderEWCharts()

          alert('✅ Chẩn đoán rủi ro thành công!')
        }
      } catch (error) {
        alert('❌ Lỗi khi kiểm tra cảnh báo: ' + (error.response?.data?.detail || error.message))
      } finally {
        isEWChecking.value = false
      }
    }

    const renderEWCharts = () => {
      if (!ewCheckResult.value) return

      // 1. Health Score Gauge
      renderHealthScoreGauge()

      // 2. Cluster Radar Chart
      renderClusterRadarChart()

      // 3. PD Projection Chart
      renderPDProjectionChart()
    }

    const renderHealthScoreGauge = () => {
      const chartDom = document.getElementById('health-score-gauge')
      if (!chartDom) return

      const myChart = echarts.init(chartDom)

      const healthScore = ewCheckResult.value.health_score
      const riskLevelColor = ewCheckResult.value.risk_level_color

      const option = {
        series: [
          {
            type: 'gauge',
            startAngle: 180,
            endAngle: 0,
            min: 0,
            max: 100,
            splitNumber: 10,
            itemStyle: {
              color: riskLevelColor
            },
            progress: {
              show: true,
              width: 30
            },
            pointer: {
              show: true,
              length: '60%',
              width: 8
            },
            axisLine: {
              lineStyle: {
                width: 30,
                color: [
                  [0.4, '#EF4444'],
                  [0.6, '#FF8C00'],
                  [0.8, '#F59E0B'],
                  [1, '#10B981']
                ]
              }
            },
            axisTick: {
              show: true
            },
            splitLine: {
              length: 15,
              lineStyle: {
                width: 2,
                color: '#999'
              }
            },
            axisLabel: {
              distance: 25,
              color: '#999',
              fontSize: 12
            },
            detail: {
              valueAnimation: true,
              formatter: '{value}',
              fontSize: 40,
              fontWeight: 'bold',
              color: riskLevelColor,
              offsetCenter: [0, '70%']
            },
            data: [
              {
                value: healthScore,
                name: 'Health Score'
              }
            ]
          }
        ]
      }

      myChart.setOption(option)
    }

    const renderClusterRadarChart = () => {
      const chartDom = document.getElementById('cluster-radar-chart')
      if (!chartDom) return

      const myChart = echarts.init(chartDom)

      const clusterInfo = ewCheckResult.value.cluster_info
      const clusterMedian = clusterInfo.cluster_median_indicators

      // Lấy 14 chỉ số hiện tại (từ indicatorsDict hoặc từ checkResult)
      let currentIndicators = {}
      if (ewCheckMode.value === 'from-predict' && indicatorsDict.value) {
        currentIndicators = indicatorsDict.value
      } else {
        // Nếu upload file, cần lấy từ backend (đã được tính)
        // Tạm thời sử dụng cluster median
        currentIndicators = clusterMedian
      }

      const indicatorNames = [
        'X_1: Biên LN gộp',
        'X_2: Biên LNTT',
        'X_3: ROA',
        'X_4: ROE',
        'X_5: Nợ/TS',
        'X_6: Nợ/VCSH',
        'X_7: TT hiện hành',
        'X_8: TT nhanh',
        'X_9: Trả lãi',
        'X_10: Trả nợ gốc',
        'X_11: Tạo tiền',
        'X_12: Vòng quay HTK',
        'X_13: Kỳ thu tiền',
        'X_14: Hiệu suất TS'
      ]

      // Tính max cho mỗi indicator (để normalize)
      const maxValues = {}
      for (let i = 1; i <= 14; i++) {
        const key = `X_${i}`
        const currentVal = currentIndicators[key] || 0
        const medianVal = clusterMedian[key] || 0
        maxValues[key] = Math.max(Math.abs(currentVal), Math.abs(medianVal), 1) * 1.5
      }

      const radarIndicators = indicatorNames.map((name, index) => {
        const key = `X_${index + 1}`
        return {
          name: name,
          max: maxValues[key]
        }
      })

      const currentValues = []
      const medianValues = []

      for (let i = 1; i <= 14; i++) {
        const key = `X_${i}`
        currentValues.push(Math.abs(currentIndicators[key] || 0))
        medianValues.push(Math.abs(clusterMedian[key] || 0))
      }

      const option = {
        title: {
          text: 'So sánh với Median của Cluster',
          left: 'center',
          textStyle: {
            fontSize: 16,
            fontWeight: 'bold',
            color: '#FF6B9D'
          }
        },
        tooltip: {
          trigger: 'item'
        },
        legend: {
          bottom: 10,
          data: ['Doanh nghiệp của bạn', 'Median của Cluster']
        },
        radar: {
          indicator: radarIndicators,
          splitNumber: 4,
          shape: 'circle',
          splitArea: {
            areaStyle: {
              color: ['rgba(255, 107, 157, 0.1)', 'rgba(255, 107, 157, 0.05)']
            }
          },
          axisLine: {
            lineStyle: {
              color: 'rgba(255, 107, 157, 0.3)'
            }
          },
          splitLine: {
            lineStyle: {
              color: 'rgba(255, 107, 157, 0.3)'
            }
          }
        },
        series: [
          {
            name: 'Chỉ số tài chính',
            type: 'radar',
            data: [
              {
                value: currentValues,
                name: 'Doanh nghiệp của bạn',
                areaStyle: {
                  color: 'rgba(255, 107, 157, 0.3)'
                },
                lineStyle: {
                  color: '#FF6B9D',
                  width: 2
                },
                itemStyle: {
                  color: '#FF6B9D'
                }
              },
              {
                value: medianValues,
                name: 'Median của Cluster',
                areaStyle: {
                  color: 'rgba(59, 130, 246, 0.2)'
                },
                lineStyle: {
                  color: '#3B82F6',
                  width: 2
                },
                itemStyle: {
                  color: '#3B82F6'
                }
              }
            ]
          }
        ]
      }

      myChart.setOption(option)
    }

    const renderPDProjectionChart = () => {
      const chartDom = document.getElementById('pd-projection-chart')
      if (!chartDom) return

      const myChart = echarts.init(chartDom)

      const pdProjection = ewCheckResult.value.pd_projection

      const xAxisData = ['Hiện tại', '3 tháng', '6 tháng', '12 tháng']

      const mildData = [
        pdProjection.current,
        pdProjection.recession_mild['3_months'],
        pdProjection.recession_mild['6_months'],
        pdProjection.recession_mild['12_months']
      ]

      const moderateData = [
        pdProjection.current,
        pdProjection.recession_moderate['3_months'],
        pdProjection.recession_moderate['6_months'],
        pdProjection.recession_moderate['12_months']
      ]

      const crisisData = [
        pdProjection.current,
        pdProjection.crisis['3_months'],
        pdProjection.crisis['6_months'],
        pdProjection.crisis['12_months']
      ]

      const option = {
        title: {
          text: 'Dự báo PD theo các kịch bản vĩ mô',
          left: 'center',
          textStyle: {
            fontSize: 16,
            fontWeight: 'bold',
            color: '#FF6B9D'
          }
        },
        tooltip: {
          trigger: 'axis',
          formatter: (params) => {
            let result = `<div style="font-weight: bold; margin-bottom: 5px;">${params[0].name}</div>`
            params.forEach(param => {
              result += `<div>${param.marker}${param.seriesName}: ${param.value.toFixed(2)}%</div>`
            })
            return result
          }
        },
        legend: {
          bottom: 10,
          data: ['🟠 Suy thoái nhẹ', '🔴 Suy thoái trung bình', '⚫ Khủng hoảng']
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '15%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: xAxisData
        },
        yAxis: {
          type: 'value',
          name: 'PD (%)',
          axisLabel: {
            formatter: '{value}%'
          }
        },
        series: [
          {
            name: '🟠 Suy thoái nhẹ',
            type: 'line',
            data: mildData,
            smooth: true,
            lineStyle: {
              color: '#F59E0B',
              width: 3
            },
            itemStyle: {
              color: '#F59E0B'
            },
            areaStyle: {
              color: 'rgba(245, 158, 11, 0.1)'
            }
          },
          {
            name: '🔴 Suy thoái trung bình',
            type: 'line',
            data: moderateData,
            smooth: true,
            lineStyle: {
              color: '#FF8C00',
              width: 3
            },
            itemStyle: {
              color: '#FF8C00'
            },
            areaStyle: {
              color: 'rgba(255, 140, 0, 0.1)'
            }
          },
          {
            name: '⚫ Khủng hoảng',
            type: 'line',
            data: crisisData,
            smooth: true,
            lineStyle: {
              color: '#EF4444',
              width: 3
            },
            itemStyle: {
              color: '#EF4444'
            },
            areaStyle: {
              color: 'rgba(239, 68, 68, 0.1)'
            }
          }
        ]
      }

      myChart.setOption(option)
    }

    const getTopFeatureImportances = () => {
      if (!ewTrainResult.value || !ewTrainResult.value.feature_importances) return {}

      const importances = ewTrainResult.value.feature_importances
      const sorted = Object.entries(importances)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)

      return Object.fromEntries(sorted)
    }

    const getSeverityLabel = (severity) => {
      const labels = {
        'critical': '🔴 Nghiêm trọng',
        'moderate': '🟡 Trung bình',
        'low': '🟢 Nhẹ'
      }
      return labels[severity] || severity
    }

    const renderMarkdown = (text) => {
      if (!text) return ''

      // Simple markdown rendering
      let html = text
        .replace(/^### (.+)$/gm, '<h4 style="color: #FF6B9D; margin-top: 1.5rem; margin-bottom: 0.5rem;">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 style="color: #FF1493; margin-top: 2rem; margin-bottom: 1rem; font-weight: 900;">$1</h3>')
        .replace(/^\*\*(.+)\*\*$/gm, '<div style="font-weight: 700; margin-top: 1rem;">$1</div>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')

      html = '<p>' + html + '</p>'
      html = html.replace(/<\/li>\n<li>/g, '</li><li>').replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')

      return html
    }

    // Mounted - Add scroll listener
    if (typeof window !== 'undefined') {
      window.addEventListener('scroll', handleScroll)
    }

    return {
      // ✅ TAB STATE
      activeTab,
      // Scroll to top
      showScrollTop,
      scrollTopPosition,
      scrollToTop,
      // Chatbot - Tab Dự báo PD
      showChatbot,
      chatMessages,
      chatInput,
      isChatLoading,
      openChatbot,
      closeChatbot,
      sendChatMessage,
      // Chatbot - Dashboard
      showDashboardChatbot,
      dashboardChatMessages,
      dashboardChatInput,
      isDashboardChatLoading,
      openDashboardChatbot,
      closeDashboardChatbot,
      sendDashboardChatMessage,
      // Training
      trainFile,
      trainFileName,
      isTraining,
      trainResult,
      // Prediction
      xlsxFile,
      xlsxFileName,
      isPredicting,
      indicators,
      indicatorsDict,
      predictionResult,
      // Gemini Analysis
      isAnalyzing,
      geminiAnalysis,
      // Export
      isExporting,
      // Dashboard - OLD
      selectedIndustry,
      isAnalyzingIndustry,
      industryAnalysis,
      industryCharts,
      // Dashboard - NEW
      isFetchingData,
      industryData,
      isShowingCharts,
      chartsData,
      briefAnalysis,
      isDeepAnalyzing,
      deepAnalysisResult,
      // Dashboard Sub-tab
      dashboardSubTab,
      // PD + Industry - NEW
      pdIndustrySelected,
      pdDataSource,
      pdXlsxFile,
      pdXlsxFileName,
      isAnalyzingPdIndustry,
      pdAnalysisIndicators,
      pdAnalysisCharts,
      pdAnalysisResult,
      // Methods
      handleTrainFile,
      trainModel,
      handleXlsxFile,
      predictFromXlsx,
      analyzeWithGemini,
      exportReport,
      getRiskClass,
      getRiskLabel,
      getLendingDecisionClass,
      getLendingDecisionIcon,
      getLendingDecisionText,
      getIndustryName,
      analyzeIndustry,
      // Dashboard - NEW Methods
      fetchIndustryData,
      showCharts,
      deepAnalyze,
      // PD + Industry - NEW Methods
      handlePdXlsxFile,
      analyzePdWithIndustry,
      // Navigate
      goToPdIndustryTab,
      // Scenario Simulation - NEW FEATURE
      scenarioDataSource,
      scenarioFile,
      scenarioFileName,
      selectedScenario,
      customRevenue,
      customInterest,
      customCogs,
      customLiquidity,
      isSimulating,
      scenarioResult,
      isAnalyzingScenario,
      scenarioAnalysis,
      showScenarioChatbot,
      scenarioChatMessages,
      scenarioChatInput,
      isScenarioChatLoading,
      isExportingScenario,
      handleScenarioFile,
      canRunSimulation,
      runScenarioSimulation,
      analyzeScenario,
      openScenarioChatbot,
      closeScenarioChatbot,
      sendScenarioChatMessage,
      exportScenarioReport,
      getPdChangeClass,
      getChangeClass,
      getChangeText,
      // Macro Scenario Simulation - NEW FEATURE
      macroDataSource,
      macroFile,
      macroFileName,
      selectedMacroScenario,
      selectedIndustryCode,
      customGdp,
      customCpi,
      customPpi,
      customPolicyRate,
      customFx,
      isSimulatingMacro,
      macroResult,
      isAnalyzingMacro,
      macroAnalysis,
      handleMacroFile,
      canRunMacroSimulation,
      runMacroSimulation,
      analyzeMacro,
      // Chatbot - Macro
      showMacroChatbot,
      macroChatMessages,
      macroChatInput,
      isMacroChatLoading,
      openMacroChatbot,
      closeMacroChatbot,
      sendMacroChatMessage,
      // Early Warning System - NEW FEATURE
      ewTrainFile,
      ewTrainFileName,
      isEWTraining,
      ewTrainResult,
      ewCheckMode,
      ewCheckFile,
      ewCheckFileName,
      ewReportPeriod,
      ewIndustryCode,
      isEWChecking,
      ewCheckResult,
      handleEWTrainFile,
      trainEarlyWarningModel,
      handleEWCheckFile,
      checkEarlyWarning,
      renderEWCharts,
      getTopFeatureImportances,
      getSeverityLabel,
      renderMarkdown,
      // Chatbot - Early Warning
      showEWChatbot,
      ewChatMessages,
      ewChatInput,
      isEWChatLoading,
      openEWChatbot,
      closeEWChatbot,
      sendEWChatMessage,
      // Anomaly Detection System - NEW FEATURE
      anomalyTrainFile,
      anomalyTrainFileName,
      isAnomalyTraining,
      anomalyTrainResult,
      anomalyDataSource,
      anomalyCheckFile,
      anomalyCheckFileName,
      isAnomalyChecking,
      anomalyCheckResult,
      canCheckAnomaly,
      handleAnomalyTrainFile,
      trainAnomalyModel,
      handleAnomalyCheckFile,
      checkAnomaly,
      renderAnomalyScoreGauge,
      renderComparisonRadarChart
    }
  }
}
</script>
