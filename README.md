# HỆ ĐA TÁC TỬ HỖ TRỢ QUYẾT ĐỊNH Y KHOA BỆNH TIỂU ĐƯỜNG
### HYBRID MULTIAGENT SYSTEM ARCHITECTURE FOR DIABETES DECISION SUPPORT
**Integrating Deep Learning, Knowledge Graph/RAG, Rule-Based Reasoning, and Reinforcement Learning**  
**Học phần:** Phát triển các hệ thống thông minh (Intelligent System Development - ISD)  
**Đơn vị:** Học viện Công nghệ Bưu chính Viễn thông (PTIT) — Khoa Công nghệ Thông tin 1  
**Thời gian:** Tháng 11/2026  

---

## 1. TỔNG QUAN VÀ MÔ HÌNH HÓA TOÁN HỌC

Hệ thống hiện thực hóa trọn vẹn kiến trúc Hệ Đa Tác Tử Lai Ghép (Hybrid Multiagent System - MAS) hỗ trợ ra quyết định lâm sàng bệnh tiểu đường dựa trên hai phân hệ nền tảng kế thừa:
1. Mô hình Học sâu (Deep Learning - DL) dự đoán tiểu đường và chuỗi thời gian CGM.
2. Cơ sở Tri thức Đồ thị (Knowledge Graph - KG) và Hệ thống RAG (Retrieval-Augmented Generation) triển khai qua Neo4j và tệp kịch bản `DB_diabet.cypher`.

### 1.1. Mô hình trừu tượng toán học của Hệ Đa Tác Tử (MAS)
Hệ thống được biểu diễn bằng bộ 5 thành phần hình thức:
$$\mathcal{M} = (\mathcal{A}, \mathcal{S}, \mathcal{O}, \mathcal{C}, \mathcal{F})$$

Trong đó:
- $\mathcal{A} = \{A_{data}, A_{DL}, A_{rule}, A_{KG}, A_{coord}, A_{exp}, A_{monitor}, A_{RL}\}$ là tập hợp 8 tác tử chuyên trách.
- $\mathcal{S}$: Không gian trạng thái toàn cục của hệ thống (Global state space).
- $\mathcal{O}$: Tập hợp các quan sát của từng tác tử (Agent observations).
- $\mathcal{C}$: Cơ chế giao tiếp qua thông điệp có cấu trúc chuẩn hóa $m_i$.
- $\mathcal{F}$: Hàm tổng hợp và điều phối kết quả:
  $$\mathcal{F}(y_{DL}, y_{rule}, y_{KG}, y_{monitor}) = y_{system}$$
  *(Hàm tổng hợp bảo toàn ranh giới minh bạch giữa dự đoán mô hình, kết quả luật và bằng chứng tri thức).*

### 1.2. Ánh xạ vào - ra của từng tác tử ($A_i: \mathcal{X}_i \to \mathcal{Y}_i$)
- $A_{data}: \mathcal{X}_{patient\_raw} \to \mathcal{Y}_{data}$ (Xác thực dữ liệu, chuẩn hóa 15 đặc trưng hoặc trả về lỗi).
- $A_{DL}: \mathcal{X}_{patient} \to \mathcal{Y}_{prediction}$ (Suy luận mô hình DL học sâu 6 tầng và chuỗi CGM 4h).
- $A_{rule}: (\mathcal{X}_{patient}, \mathcal{Y}_{prediction}) \to \mathcal{Y}_{rule}$ (Suy luận luật, cờ kiểm toán, kiểm định liều thuốc theo eGFR).
- $A_{KG}: \mathcal{Q} \to \mathcal{E}$ (Truy xuất bằng chứng đồ thị và văn bản).
- $A_{RL}: \mathcal{S}_{sim} \to \mathcal{A}_{sim}$ (Học chính sách giáo dục / theo dõi trong môi trường mô phỏng tổng hợp).
- $A_{exp}: (\mathcal{Y}_{prediction}, \mathcal{Y}_{rule}, \mathcal{E}) \to \mathcal{Y}_{report}$ (Sinh báo cáo có cấu trúc: Dự đoán + Bằng chứng + Giới hạn).
- $A_{monitor}: \text{Metadata} \to \text{AuditRecord}$ (Ghi vết kiểm toán và hiệu năng toàn trình).

---

## 2. KIẾN TRÚC HỆ THỐNG 6 TẦNG (LAYERED ARCHITECTURE)

Hệ thống được tổ chức phân tầng nghiêm ngặt thành 6 tầng kiến trúc độc lập và tương tác tuần tự:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               TẦNG 1: GIAO DIỆN NGƯỜI DÙNG (USER INTERACTION)                │
│    (Bác sĩ / Bệnh nhân / Sinh viên nghiên cứu - Terminal CLI & Streamlit)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            TẦNG 2: CỔNG API & TIỀN XỬ LÝ (API & DATA PROCESSING)             │
│   (Tiếp nhận yêu cầu, PatientDataAgent kiểm thực kiểu dữ liệu, ngưỡng sinh lý)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│          TẦNG 3: ĐIỀU PHỐI ĐA TÁC TỬ (MULTIAGENT COORDINATION)               │
│   (CoordinatorAgent điều phối thuật toán 12 bước logic, gom & tổng hợp kết quả)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                TẦNG 4: TÁC TỬ CHUYÊN MÔN (INTELLIGENT AGENTS)                │
│  [A_data] ──► [A_DL] ──► [A_rule] ──► [A_KG] ──► [A_exp] ──► [A_monitor]    │
│                                       └─► [A_RL] (Mô phỏng tổng hợp)         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            TẦNG 5: TRI THỨC VÀ MÔ HÌNH (KNOWLEDGE & MODEL LAYER)             │
│ (Mô hình DL 6 tầng, Neo4j DB_diabet.cypher, Vector RAG, Môi trường MDP mô phỏng)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            TẦNG 6: ĐÁNH GIÁ VÀ KIỂM TOÁN (EVALUATION & AUDIT LAYER)          │
│ (CSDL SQLite medical_intelligent_system.db: agent_action_logs, Groundedness)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. PHÂN LOẠI VÀ GIAO DIỆN HỢP ĐỒNG 8 TÁC TỬ

| Tác tử | Ký hiệu | Mô hình AI | Đầu vào (Input) | Đầu ra (Output) | Vai trò & Trách nhiệm chính |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Patient Data Agent** | $A_{data}$ | Xử lý tất định | Hồ sơ bệnh nhân thô | Dữ liệu chuẩn hóa hoặc lỗi | Xác thực kiểu dữ liệu, bắt lỗi giá trị thiếu, kiểm tra ngưỡng sinh lý, phòng chống rò rỉ dữ liệu. |
| **DL Prediction Agent** | $A_{DL}$ | Deep Learning | 15 đặc trưng chuẩn hóa | Xác suất, nhãn, dự báo CGM | Chạy mạng 6 tầng $f_6 \circ \dots \circ f_1$ và RNN dự báo đường huyết chuỗi thời gian 4h tới. |
| **Rule-Based Agent** | $A_{rule}$ | Suy luận ký hiệu | Đặc trưng & Dự đoán DL | Kết quả luật, cờ kiểm toán | Áp dụng quy tắc QĐ 5481 & ADA 2024, kiểm tra cấp cứu, kiểm định chống chỉ định Metformin theo eGFR, phát hiện mâu thuẫn. |
| **KG/RAG Agent** | $A_{KG}$ | Knowledge Graph & RAG | Câu truy vấn tự nhiên | Bằng chứng đồ thị & văn bản | Thực hiện 6 bước: Diễn giải -> Nhận diện thực thể -> Duyệt Neo4j -> Vector RAG -> Xếp hạng -> Quy gán nguồn gốc. |
| **Coordinator Agent** | $A_{coord}$ | Điều phối đa tác tử | Yêu cầu người dùng | Kết quả hội chẩn tổng hợp | Lập lịch tuần tự 12 bước, gom kết quả theo hàm $\mathcal{F}$, xử lý lỗi xác thực sớm. |
| **Explanation Agent** | $A_{exp}$ | Template/LLM Reasoning | Dự đoán, Luật, Bằng chứng | Báo cáo hỗ trợ quyết định | Xuất Báo cáo Hỗ trợ Quyết định (Dự đoán + Bằng chứng + Giới hạn), kiểm soát ảo giác y khoa. |
| **Monitoring Agent** | $A_{monitor}$ | Giám sát & Kiểm toán | Siêu dữ liệu thực thi | Bản ghi kiểm toán (Audit) | Theo dõi vết thực thi, phiên bản tác tử, đo đạc độ trễ toàn trình, ghi vào bảng SQLite `agent_action_logs`. |
| **RL Agent (Mô phỏng)** | $A_{RL}$ | Reinforcement Learning | Trạng thái mô phỏng $s$ | Hành động mô phỏng $a$ | Học chính sách tối ưu Bellman $\pi^*$ trong môi trường MDP tổng hợp (giáo dục lối sống, chu kỳ theo dõi). |

---

## 4. HỢP ĐỒNG DỮ LIỆU & THÔNG ĐIỆP CHUẨN HÓA ($m_i$)

Mọi tác tử trao đổi dữ liệu qua cấu trúc thông điệp chuẩn mực quy định tại **Mục 6.1 & 6.2**:
$$m_i = (id, agent, status, payload, evidence, version)$$

### Ví dụ thông điệp JSON thực tế ($A_{DL}$):
```json
{
  "request_id": "req_de881705",
  "agent": "DiabetesPredictionAgent",
  "status": "ok",
  "payload": {
    "prediction": 1,
    "score": 0.9767,
    "probability": 0.9767,
    "risk_level": "NGUY CƠ RẤT CAO (Bệnh nhân ĐTĐ Típ 2)",
    "cgm_forecast": {
      "forecast_4h": [198.7, 200.3, 201.4, 202.3],
      "peak_glucose": 202.3
    }
  },
  "evidence": [
    "Suy luận mạng 6 tầng (15 -> 64 -> 64 -> 32 -> 16 -> 8 -> 1)",
    "Dự báo chuỗi CGM bằng mạng hồi quy RNN dừng sai phân bậc 1"
  ],
  "model_version": "diabetes_dl_v1"
}
```

---

## 5. QUY TRÌNH ĐIỀU PHỐI TUẦN TỰ 12 BƯỚC (COORDINATOR WORKFLOW)

Quy trình điều phối trung tâm được Coordinator Agent thực thi theo thuật toán Mục 5.3:
1. **Tiếp nhận yêu cầu:** Nhận hồ sơ lâm sàng bệnh nhân và câu truy vấn.
2. **Sinh định danh duy nhất:** Tạo `request_id` (ví dụ `req_xxxxxxxx`).
3. **Gọi PatientDataAgent ($A_{data}$):** Kiểm tra cấu trúc, kiểu dữ liệu, các trường bắt buộc và dải đo sinh lý.
4. **Kiểm tra trạng thái xác thực:** Nếu `validation_passed == False`, chuyển ngay sang Bước 5.
5. **Chấm dứt sớm khi có lỗi (Fail-Fast):** Trả về mã lỗi chi tiết và kết thúc quy trình sớm nhằm bảo đảm an toàn y tế.
6. **Gọi DLPredictionAgent ($A_{DL}$):** Chạy mạng 6 tầng và mạng RNN dự báo CGM 4h.
7. **Gọi RuleBasedAgent ($A_{rule}$):** Rà soát tiêu chuẩn ADA/BYT, kiểm tra phản xạ cấp cứu, và hiệu chỉnh liều Metformin theo eGFR.
8. **Gọi KnowledgeRetrievalAgent ($A_{KG}$):** Duyệt 2-hop trên Neo4j (`DB_diabet.cypher`) và truy xuất văn bản Vector RAG.
9. **Thu thập kết quả & Kích hoạt RL Agent:** Lấy kết quả từ các tác tử và tra cứu chính sách giáo dục mô phỏng từ Q-Table.
10. **Gọi ExplanationAgent ($A_{exp}$):** Tổng hợp thành Báo cáo Hỗ trợ Quyết định (Dự đoán + Bằng chứng + Giới hạn).
11. **Ghi vết kiểm toán qua MonitoringAgent ($A_{monitor}$):** Lưu toàn bộ vết tương tác vào CSDL SQLite `agent_action_logs`.
12. **Trả về kết quả có cấu trúc:** Xuất đối tượng $y_{system}$ cho giao diện người dùng.

---

## 6. ĐẶC TẢ TÍCH HỢP MÔ HÌNH HỌC SÂU (MỤC 7.1)

Đáp ứng đầy đủ 7 tiêu chí bắt buộc theo đặc tả kỹ thuật:
1. **Kiến trúc mô hình:** Mạng nơ-ron học sâu 6 tầng tham số $15 \to 64 \to 64 \to 32 \to 16 \to 8 \to 1$ kết hợp mạng RNN dừng $\Delta g_t$.
2. **Tập dữ liệu huấn luyện:** Kaggle Diabetes Healthcare Dataset (100,000 bản ghi) kết hợp PTIT Clinical Cohort.
3. **Thứ tự 15 đặc trưng đầu vào:** `Age`, `Gender_Code`, `BMI`, `HbA1c`, `Fasting_Glucose`, `Systolic_BP`, `Diastolic_BP`, `eGFR`, `Glucose_x_BMI`, `HbA1c_x_Age`, `MAP`, `Glucose_to_eGFR`, `Hypertension`, `HeartDisease`, `SmokingScore`.
4. **Phương pháp tiền xử lý:** Chuẩn hóa Z-score bất biến $z = (x - \mu)/\sigma$, phòng chống rò rỉ dữ liệu.
5. **Định dạng đầu ra:** Xác suất $\hat{y} \in [0, 1]$, nhãn phân loại $\{0, 1\}$, vector Bottleneck 8 chiều $h^{(5)}$ và chuỗi dự báo CGM 4h.
6. **Phiên bản mô hình:** `diabetes_dl_v1`.
7. **Các chỉ số kiểm định:** Accuracy = 96.2%, F1-Score = 0.958, AUC-ROC = 0.984, Sensitivity = 94.7%, Specificity = 96.8%.
> [!WARNING]
> **Lưu ý Giới hạn Lâm sàng (Mục 7.1):** Hệ thống không mặc định mô hình đã được xác thực lâm sàng chỉ vì nó đạt độ chính xác cao trên tập kiểm nghiệm.

---

## 7. ĐỒ THỊ TRI THỨC NEO4J & DB_DIABET.CYPHER (MỤC 7.2)

- Kho tri thức đồ thị được khởi tạo từ tệp script chính thức **`DB_diabet.cypher`** (và bản sao tương thích `kb_diabet_neo4j.cypher`), chứa 33 nút thực thể và 32 quan hệ ngữ nghĩa y khoa.
- Tác tử $A_{KG}$ thực hiện tuần tự 6 bước:
  1. *Query interpretation:* Phân tích ngữ nghĩa yêu cầu.
  2. *Entity identification:* Nhận diện thực thể bệnh lý, chỉ số sinh học, dược lý.
  3. *Graph retrieval:* Duyệt đường dẫn 1-hop và 2-hop trên Neo4j.
  4. *Relevant document retrieval:* Tìm kiếm Top-3 đoạn văn bản bằng Cosine Similarity trên Vector Store.
  5. *Evidence ranking:* Xếp hạng các chuỗi chứng cứ y khoa.
  6. *Source attribution:* Quy gán rõ nguồn gốc tới Quyết định 5481/QĐ-BYT, Tiêu chuẩn ADA 2024 và Bệnh viện Vinmec.

---

## 8. MỞ RỘNG HỌC TĂNG CƯỜNG TRONG MÔ PHỎNG (MỤC 8)

- Mô hình hóa bài toán theo Markov Decision Process (MDP):
  $$\mathcal{M}_{RL} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$
  với không gian trạng thái $s = (g_t, c_t) \in \{0, 1\}^2$, hành động $a \in \{0, 1\}$, và hệ số chiết khấu $\gamma = 0.90$.
- **Ràng buộc an toàn tuyệt đối:**
  - Tác tử RL vận hành ĐỘC QUYỀN trong môi trường mô phỏng tổng hợp nhằm mục đích nghiên cứu và giáo dục lối sống.
  - Tác tử RL **tuyệt đối không được phép tự động kê đơn thuốc, thay đổi phác đồ hay điều trị trực tiếp cho bệnh nhân thực tế.**

---

## 9. TUYÊN BỐ AN TOÀN VÀ GIỚI HẠN NGHIÊN CỨU (MỤC 10)

Hệ thống là nguyên mẫu nghiên cứu và phục vụ học thuật (Educational and Research Prototype), tuân thủ 7 nguyên tắc an toàn:
1. Đầu ra của mô hình không phải là chẩn đoán y khoa xác nhận.
2. Hệ thống không tự ý kê đơn hoặc hiệu chỉnh liều lượng thuốc.
3. Tác tử RL chỉ hoạt động trong môi trường mô phỏng.
4. Mọi quy tắc lâm sàng phải được thẩm định bởi chuyên gia y tế có chứng chỉ.
5. Sử dụng dữ liệu tổng hợp hoặc dữ liệu công khai đã khử định danh hợp lệ.
6. Nội dung truy xuất từ KG/RAG phải truy vết minh bạch đến nguồn gốc văn bản.
7. Khi có kết quả mâu thuẫn (Conflict), hệ thống lập tức gắn cờ yêu cầu bác sĩ thẩm định (`FLAG_HUMAN_REVIEW_REQUIRED`).

---

## 10. HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH

### Bước 1: Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy kiểm thử tự động trên Terminal (Chế độ CLI)
```bash
python diabetes_multi_agent_app.py --cli
```
*Lệnh sẽ tự động chạy 2 thực nghiệm:*
- **Thực nghiệm 1:** Ca bệnh chuẩn BN001 qua quy trình 12 bước, in vết trao đổi thông điệp $m_i$ và xuất Báo cáo Hỗ trợ Quyết định (Dự đoán + Bằng chứng + Giới hạn).
- **Thực nghiệm 2:** Ca bệnh có dữ liệu lỗi, chứng minh cơ chế Fail-Fast của Coordinator Agent tại Bước 4-5.

### Bước 3: Khởi chạy Giao diện Web Dashboard (Streamlit)
```bash
streamlit run diabetes_multi_agent_app.py
```
Giao diện trực quan sẽ hiển thị tại: `http://localhost:8501`.

---

## 11. CẤU TRÚC THƯ MỤC DỰ ÁN

```
ISD_Project/
├── DB_diabet.cypher                  # Kịch bản nạp đồ thị Neo4j chuẩn theo đặc tả đề tài
├── diabetes_multi_agent_app.py       # Điểm vào chính: hỗ trợ cả --cli và giao diện Streamlit
├── extract_vinmec_kg.py              # Pipeline trích xuất tri thức bệnh viện Vinmec
├── train_models.py                   # Huấn luyện mạng học sâu và Q-Learning trong mô phỏng
├── requirements.txt                  # Danh mục thư viện phụ thuộc
├── README.md                         # Tài liệu đặc tả kiến trúc toàn trình
│
├── docs/                             # Tài liệu học tập & báo cáo học phần PTIT
│   ├── ISD_Bao_Cao_Project_He_Da_Agent_Tieu_Duong.docx   # Báo cáo toàn trình đề tài
│   ├── ISD_Tieu_Luan_Ly_Thuyet.docx                      # Tiểu luận lý thuyết hệ chuyên gia
│   └── Ngan_Hang_Cau_Hoi_Van_Dap_ISD_PTIT_50_Cau.docx    # Ngân hàng 50 câu hỏi vấn đáp
│
├── src/
│   ├── __init__.py
│   ├── database.py                   # CSDL SQLite (bảng patients, cgm_stream, agent_action_logs)
│   ├── deep_learning.py              # Mạng 6 tầng hợp hàm, RNN CGM, get_model_metadata()
│   ├── knowledge_graph.py            # Đồ thị NetworkX / Neo4j driver, duyệt 2-hop, nạp Cypher
│   ├── vector_rag.py                 # Phân đoạn trượt, TF-IDF Cosine Similarity, Hợp nhất ngữ cảnh
│   ├── rl_environment.py             # Môi trường MDP mô phỏng, Q-Table, phương trình Bellman
│   └── multi_agent_system.py         # 8 Tác tử chuẩn hóa, Hợp đồng m_i, Thuật toán điều phối 12 bước
│
├── data/
│   ├── medical_knowledge_triples.csv # Bộ ba tri thức y khoa (Subject, Relation, Object)
│   ├── guidelines_byt_5481.txt       # Hướng dẫn Quyết định 5481/QĐ-BYT Bộ Y tế Việt Nam
│   ├── guidelines_ada_2024.txt       # Tiêu chuẩn ADA Standards of Care 2024 (Hoa Kỳ)
│   ├── vinmec_diabetes_article.txt   # Tri thức đái tháo đường típ 2 Bệnh viện Vinmec
│   ├── vinmec_diabetes_triples.csv   # Bộ ba tri thức trích xuất từ bài viết Vinmec
│   └── kg_graph.json                 # Đồ thị tri thức định dạng JSON chuẩn
│
└── lib/                              # Thư viện trực quan hóa vis-network và UI assets
```
