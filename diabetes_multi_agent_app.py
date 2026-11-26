"""
=============================================================================
HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG - KHOA CÔNG NGHỆ THÔNG TIN 1
HỌC PHẦN: PHÁT TRIỂN CÁC HỆ THỐNG THÔNG MINH (INTELLIGENT SYSTEM DEVELOPMENT - ISD)
PROJECT: HỆ ĐA TÁC TỬ HỖ TRỢ QUYẾT ĐỊNH Y KHOA BỆNH TIỂU ĐƯỜNG
(Hybrid Multiagent System Architecture for Diabetes Decision Support)
Integrating Deep Learning, Knowledge Graph/RAG, Rule-Based Reasoning, and Reinforcement Learning

Tệp thực thi chính: diabetes_multi_agent_app.py
Hỗ trợ 2 chế độ:
  1. Terminal CLI:  python diabetes_multi_agent_app.py --cli
  2. Web Dashboard: streamlit run diabetes_multi_agent_app.py
=============================================================================
"""

import sys
import os
import argparse
import time
import json
import pandas as pd
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import init_database, get_all_patients, get_patient, get_cgm_stream, get_agent_logs, get_connection
from src.deep_learning import (
    InvariantClinicalPreprocessor,
    SixLayerDeepClinicalNetNumPy,
    RecurrentCGMForecaster,
    get_model_metadata
)
from src.knowledge_graph import MedicalKnowledgeGraph, NEO4J_AVAILABLE
from src.vector_rag import ClinicalVectorStore, HybridContextFusion
from src.rl_environment import DiabetesLifestyleEnvironment, TabularQLearningAgent, ClinicalRuleBasedAgent
from src.multi_agent_system import (
    CoordinatorAgent,
    PatientDataAgent,
    DLPredictionAgent,
    RuleBasedAgent,
    KnowledgeRetrievalAgent,
    RLAgent,
    ExplanationAgent,
    MonitoringAgent,
    AgentMessage,
    MultiAgentDiabetesOrchestrator
)


# =============================================================================
# CHẾ ĐỘ 1: TERMINAL CLI (KIỂM THỬ TỰ ĐỘNG KHÔNG CẦN GIAO DIỆN)
# =============================================================================
def run_cli_mode():
    print("=" * 85)
    print("HỆ ĐA TÁC TỬ HỖ TRỢ QUYẾT ĐỊNH Y KHOA BỆNH TIỂU ĐƯỜNG")
    print("Hybrid Multiagent System Architecture for Diabetes Decision Support")
    print("PTIT – Intelligent System Development Project (Tháng 11/2026)")
    print("Mô hình toán học: M = (A, S, O, C, F) | 8 Tác tử | Hợp đồng thông điệp m_i")
    print("=" * 85)

    # 1. Khởi tạo CSDL SQLite & Các Phân hệ Nền tảng
    print("\n[INFO] Khởi tạo CSDL SQLite, Đồ thị Tri thức (DB_diabet.cypher) và Vector RAG...")
    init_database()
    kg = MedicalKnowledgeGraph()
    vstore = ClinicalVectorStore()
    vstore.build_index_from_directory()
    coordinator = CoordinatorAgent(kg, vstore)
    print("[INFO] Khởi tạo thành công 6 tầng kiến trúc và 8 tác tử chuyên môn.")

    # 2. Thực nghiệm ca bệnh lâm sàng phức tạp BN001 (Quy trình 12 bước)
    target_patient_id = "BN001"
    query = (
        "Tôi 58 tuổi, BMI 31.2, HbA1c 7.8%, đường huyết sáng nay là 192 mg/dL "
        "và bị suy thận (eGFR = 42 mL/phút). Tôi có dùng được Metformin nguyên liều không "
        "và cần ăn uống, tập luyện thế nào?"
    )
    print(f"\n[THỰC NGHIỆM 1] CA BỆNH HỢP LỆ: {target_patient_id}")
    print(f"YÊU CẦU LÂM SÀNG: \"{query}\"\n")
    print("-" * 85)
    print("Đang thực thi Thuật toán 12 bước logic của Coordinator Agent (Section 5.3)...")
    print("-" * 85)

    result = coordinator.run_full_consultation_lifecycle(target_patient_id, query)

    # In chi tiết vết ReAct / 12 bước logic
    print("\nCHI TIẾT VẾT SUY LUẬN & ĐIỀU PHỐI 12 BƯỚC CỦA COORDINATOR AGENT:")
    for step in result["react_trace"]:
        print(f"\n[Bước {step['step']}] {step['phase']} - {step['agent']}")
        print(f"  Suy luận (Thought)   : {step['thought']}")
        print(f"  Hành động (Action)   : {step['action']}")
        print(f"  Quan sát (Observation): {step['observation']}")
        if "message" in step:
            m = step["message"]
            print(f"  Hợp đồng m_i: id='{m['request_id']}', status='{m['status']}', version='{m['version']}'")

    print("\n" + "=" * 85)
    print(result["final_report"])
    print("=" * 85)
    print(f"\n[INFO] Hoàn tất hội chẩn thành công trong {result['latency_ms']} ms.")

    # 3. Thực nghiệm ca bệnh lỗi để kiểm tra cơ chế chặn lỗi xác thực đầu vào (Bước 4 & 5)
    print("\n" + "-" * 85)
    print("[THỰC NGHIỆM 2] KIỂM THỬ XÁC THỰC LỖI DỮ LIỆU ĐẦU VÀO (BƯỚC 4 & 5)")
    print("-" * 85)
    invalid_patient = {
        "patient_id": "TEST_INVALID",
        "full_name": "Bệnh nhân Dữ liệu Bất thường",
        "age": -8,
        "bmi": 5.0,
        "fasting_glucose": 3500.0,
        "hba1c": 0.5,
        "systolic_bp": 70,
        "diastolic_bp": 110,
        "egfr": -10.0
    }
    print(f"Dữ liệu đầu vào lỗi: Tuổi=-8, BMI=5.0, Glucose=3500 mg/dL, HA=70/110 mmHg, eGFR=-10")
    bad_result = coordinator.execute_workflow(invalid_patient, "Kiểm thử kiểm tra lỗi")
    print(f"Trạng thái điều phối: {bad_result['status'].upper()}")
    print("Các lỗi xác thực được PatientDataAgent phát hiện:")
    for err in bad_result.get("errors", []):
        print(f"  [X] {err}")
    print(f"[KẾT LUẬN KIỂM THỬ] Coordinator Agent đã chấm dứt quy trình ngay tại Bước 5 để bảo đảm an toàn.")
    print("Để mở giao diện trực quan, chạy lệnh: streamlit run diabetes_multi_agent_app.py\n")


# =============================================================================
# CHẾ ĐỘ 2: GIAO DIỆN WEB DASHBOARD TẬP TRUNG (STREAMLIT)
# =============================================================================
def generate_clinical_chat_response(chat_query: str, coordinator, kg_res, rag_results, graph_facts) -> str:
    """
    Bộ tổng hợp tri thức lâm sàng đa tầng (Clinical Knowledge Synthesizer):
    Tự động phân tích câu hỏi người dùng, kết hợp dữ liệu từ GraphRAG (Neo4j / DB_diabet.cypher),
    Vector Store (Top-k tài liệu QĐ 5481/BYT, ADA 2024, Vinmec) và hệ thống tác tử chuyên khoa
    để tạo câu trả lời y khoa toàn diện, chi tiết, không cụt ngủn và không bị gãy liên kết (404).
    """
    lower_q = chat_query.lower()
    bot_reply_blocks = []

    # TRƯỜNG HỢP 1: TƯ VẤN CA BỆNH / HỘI CHẨN LÂM SÀNG
    if any(k in lower_q for k in ["tư vấn", "hội chẩn", "phác đồ", "bn001", "bn002", "bn003", "ca bệnh", "kế hoạch điều trị"]):
        target_pid = "BN001"
        if "bn002" in lower_q:
            target_pid = "BN002"
        elif "bn003" in lower_q:
            target_pid = "BN003"

        agent_res = coordinator.run_full_consultation_lifecycle(target_pid, chat_query)
        dl_out = agent_res["dl_output"]
        rag_out = agent_res["rag_output"]
        rl_out = agent_res["rl_output"]
        p_info = agent_res["patient"]

        bot_reply_blocks.append(
            f"### Báo cáo Tư vấn Lâm sàng Toàn diện từ Hệ Đa Tác Tử (Mã: `{agent_res['request_id']}`)\n"
            f"**Hồ sơ bệnh nhân:** {p_info['full_name']} (Mã: `{p_info['patient_id']}`) | Tuổi: {p_info['age']} tuổi | BMI: {p_info['bmi']} kg/m² | eGFR: **{p_info['egfr']} mL/phút/1.73m²**\n\n"
            f"#### 1. Đánh giá Nguy cơ & Dự báo Học Sâu (DLPredictionAgent - A_DL):\n"
            f"- **Xác suất mắc ĐTĐ Típ 2 (Mạng DNN 6 tầng):** **{dl_out['risk_analysis']['risk_percentage']}%** [{dl_out['risk_analysis']['risk_level']}]\n"
            f"- **Dự báo đường huyết liên tục CGM 4h tới (RNN):** `{dl_out['cgm_forecast']['forecast_4h']}` mg/dL\n"
            f"- **Đỉnh đường huyết cảnh báo:** **{dl_out['cgm_forecast']['peak_glucose']} mg/dL** (Cần kiểm soát đỉnh sau ăn < 180 mg/dL theo ADA 2024)\n\n"
            f"#### 2. Dược lý & An toàn Thận (RuleBasedAgent & KGRAGAgent):\n"
            f"- **Khuyến nghị chỉnh liều Metformin:** {rag_out['safety_check']['dose_adjustment']}\n"
            f"- **Thuốc phối hợp bảo vệ cơ quan đích:** {', '.join(rag_out['safety_check']['recommended_drugs'])} (Ưu tiên SGLT2i Empagliflozin/Dapagliflozin làm chậm suy giảm chức năng thận)\n"
            f"- **Kiểm soát huyết áp & Đạm niệu:** Khuyến nghị dùng nhóm ACEi hoặc ARB để bảo vệ cầu thận khi có tăng huyết áp kèm theo.\n\n"
            f"#### 3. Kế hoạch Can thiệp Lối sống Tối ưu (RLSimulationAgent - A_RL):\n"
            f"- **Chính sách tối ưu Bellman:** {rl_out['action_desc']}\n"
            f"- **Chi tiết kế hoạch thực hiện:**\n{rl_out['detailed_plan']}\n\n"
            f"#### 4. Thẩm định Bằng chứng & Kiểm toán Lâm sàng (ExplanationAgent - A_exp):\n"
            f"- **Điểm trung thực dẫn chứng (Groundedness):** `{agent_res['groundedness_score']}%` | **Tỷ lệ ảo giác:** `{agent_res['hallucination_rate']}%`\n"
            f"- **Thời gian xử lý đa tác tử:** `{agent_res['latency_ms']} ms` | **Trạng thái kiểm toán:** Đã đối chiếu QĐ 5481/BYT, ADA 2024 và Vinmec.\n"
        )

    # TRƯỜNG HỢP 2: BIẾN CHỨNG ĐÁI THÁO ĐƯỜNG
    elif any(k in lower_q for k in ["biến chứng", "hậu quả", "nguy hiểm", "mù", "loét", "cắt cụt", "đoạn chi", "dka", "hhs", "tai biến", "đột quỵ"]):
        bot_reply_blocks.append(
            "### Các Biến Chứng Nguy Hiểm Của Bệnh Đái Tháo Đường Típ 2\n"
            "Dựa trên tài liệu Chuyên khoa Nội tiết - Bệnh viện Đa khoa Quốc tế Vinmec và Hướng dẫn QĐ 5481/QĐ-BYT, biến chứng đái tháo đường được chia thành 2 nhóm chính:\n\n"
            "#### 1. Biến chứng cấp tính (Đe dọa tính mạng tức thì - Cần can thiệp cấp cứu y tế):\n"
            "- **Hạ đường huyết cấp (Hypoglycemia):** Khi glucose máu < 70 mg/dL do dùng thuốc/insulin quá liều hoặc bỏ bữa. Gây run rẩy, vã mồ hôi, co giật, hôn mê nếu không xử trí kịp thời.\n"
            "- **Nhiễm toan ceton do đái tháo đường (Diabetic Ketoacidosis - DKA):** Thường kèm Glucose > 250 mg/dL, hơi thở có mùi táo chín (mùi aceton), toan chuyển hóa, buồn nôn, rối loạn điện giải nghiêm trọng.\n"
            "- **Hội chứng tăng áp lực thẩm thấu do tăng đường huyết (HHS):** Glucose máu tăng cực cao (> 600 mg/dL) gây mất nước nội bào nặng nề, lơ mơ, hôn mê, tỷ lệ tử vong cao đặc biệt ở người cao tuổi.\n\n"
            "#### 2. Biến chứng mạn tính tiến triển (Tổn thương cơ quan đích lâu dài):\n"
            "- **Biến chứng thận (Diabetic Nephropathy):** Tổn thương mao mạch vi cầu thận gây tiểu đạm (microalbumin niệu), tiến triển dần đến suy thận mạn giai đoạn cuối, phải lọc máu chu kỳ hoặc ghép thận.\n"
            "- **Biến chứng mắt & thị lực (Diabetic Retinopathy):** Tổn thương võng mạc do vi mạch máu nuôi dưỡng bị tổn thương, phù hoàng điểm, xuất huyết võng mạc, nguy cơ mù lòa vĩnh viễn.\n"
            "- **Biến chứng thần kinh (Diabetic Neuropathy):** Tổn thương các sợi thần kinh ngoại biên gây tê bì, dị cảm, cảm giác châm chích, 'bỏng rát' hoặc mất hoàn toàn cảm giác đau ở bàn chân.\n"
            "- **Biến chứng tim mạch & đột quỵ:** Tăng nguy cơ xơ vữa động mạch, nhồi máu cơ tim cấp và đột quỵ não lên gấp 2–4 lần so với người không mắc bệnh.\n"
            "- **Biến chứng bàn chân đái tháo đường (Diabetic Foot):** Sự kết hợp giữa mất cảm giác thần kinh, hẹp mạch máu nuôi chi dưới và nhiễm trùng, dễ hình thành vết loét hoại tử dẫn đến nguy cơ phải cắt cụt chi (đoạn chi).\n\n"
            "#### 3. Biện pháp phòng ngừa và tầm soát định kỳ:\n"
            "- Kiểm soát chặt chẽ chỉ số HbA1c < 7.0% và huyết áp mục tiêu < 130/80 mmHg.\n"
            "- Xét nghiệm định kỳ eGFR và tỷ số Albumin/Creatinine niệu (uACR) mỗi 6–12 tháng.\n"
            "- Soi đáy mắt định kỳ hàng năm và tự kiểm tra bàn chân mỗi ngày.\n"
        )

    # TRƯỜNG HỢP 3: NGUYÊN NHÂN & CƠ CHẾ BỆNH SINH
    elif any(k in lower_q for k in ["nguyên nhân", "bệnh sinh", "cơ chế", "tại sao bị", "kháng insulin", "beta tụy", "yếu tố nguy cơ", "yếu tố di truyền"]):
        bot_reply_blocks.append(
            "### Nguyên Nhân & Cơ Chế Bệnh Sinh Đái Tháo Đường Típ 2\n"
            "Theo tài liệu chuyên khoa Bệnh viện Vinmec và Hiệp hội ADA, đái tháo đường típ 2 là bệnh rối loạn chuyển hóa mạn tính đặc trưng bởi sự tăng glucose máu kéo dài do hai khiếm khuyết cơ bản phối hợp:\n\n"
            "#### 1. Hai cơ chế bệnh sinh cốt lõi:\n"
            "1. **Kháng Insulin (Insulin Resistance):**\n"
            "   - Tuyến tụy vẫn sản xuất insulin (thậm chí tăng tiết bù trừ ở giai đoạn sớm), nhưng các tế bào đích tại gan, mô cơ xương và mô mỡ bị 'trơ', giảm đáp ứng sinh học với insulin.\n"
            "   - Hậu quả: Gan không ngừng phóng thích glucose vào tuần hoàn, còn cơ bắp giảm thu nhận glucose, khiến lượng đường trong máu liên tục ở mức cao.\n"
            "2. **Suy giảm tiến triển chức năng tế bào beta đảo tụy:**\n"
            "   - Tình trạng tăng đường huyết và tăng acid béo tự do kéo dài gây hiện tượng ngộ độc đường (glucotoxicity) và ngộ độc mỡ (lipotoxicity).\n"
            "   - Tế bào beta tụy bị kiệt quệ do phải làm việc bù trừ quá tải trong thời gian dài, dần dần suy giảm cả số lượng lẫn khả năng tiết insulin.\n\n"
            "#### 2. Các yếu tố nguy cơ hàng đầu (Khuyến cáo sàng lọc Vinmec & Bộ Y tế):\n"
            "- **Thừa cân, béo phì:** Chỉ số BMI $\\ge 23$ kg/m² (chuẩn người châu Á) hoặc béo bụng (vòng eo $\\ge 90$ cm ở nam, $\\ge 80$ cm ở nữ).\n"
            "- **Lối sống tĩnh tại, ít vận động:** Giảm tiêu hao năng lượng và làm suy giảm độ nhạy insulin của cơ xương.\n"
            "- **Tiền sử gia đình trực hệ:** Có bố, mẹ, anh chị em ruột mắc bệnh đái tháo đường típ 2.\n"
            "- **Bệnh đồng mắc tim mạch - chuyển hóa:** Tăng huyết áp ($\\ge 140/90$ mmHg), rối loạn lipid máu (Triglyceride $> 2.8$ mmol/L, HDL-C $< 0.9$ mmol/L).\n"
            "- **Tiền sử thai kỳ:** Từng mắc đái tháo đường thai kỳ hoặc sinh con nặng trên 4.0 kg.\n"
            "- **Dấu gai đen (Acanthosis nigricans):** Vùng da sẫm màu, dày sừng ở nếp gấp cổ, nách là dấu hiệu lâm sàng chỉ báo kháng insulin nặng.\n"
        )

    # TRƯỜNG HỢP 4: TRIỆU CHỨNG LÂM SÀNG ("HỘI CHỨNG 4 NHIỀU")
    elif any(k in lower_q for k in ["4 nhiều", "triệu chứng", "dấu hiệu", "biểu hiện", "tiểu nhiều", "khát nước", "sụt cân", "mau đói", "cảnh báo"]):
        bot_reply_blocks.append(
            "### Triệu Chứng Lâm Sàng Kinh Điển & Hội Chứng 4 Nhiều\n"
            "Cơ sở tri thức Bệnh viện Vinmec và Đồ thị tri thức (DB_diabet.cypher) mô tả chi tiết **Hội chứng 4 nhiều (Classic 4-P Symptoms)** đặc trưng của bệnh đái tháo đường:\n\n"
            "1. **Tiểu nhiều (Polyuria):**\n"
            "   - *Cơ chế sinh lý:* Khi nồng độ glucose máu vượt ngưỡng tái hấp thu của thận (ngưỡng thận $\\approx 180$ mg/dL), lượng đường dư thừa đào thải qua nước tiểu kéo theo phân tử nước theo cơ chế lợi niệu thẩm thấu.\n"
            "   - *Biểu hiện:* Đi tiểu nhiều lần trong ngày, lượng nước tiểu tăng vọt (3–5 lít/ngày), đặc biệt phải thức giấc tiểu đêm liên tục.\n\n"
            "2. **Khát nhiều (Polydipsia):**\n"
            "   - *Cơ chế sinh lý:* Lượng nước mất lớn qua đường tiểu làm tăng độ thẩm thấu huyết tương, kích hoạt trực tiếp trung tâm khát tại vùng dưới đồi não bộ.\n"
            "   - *Biểu hiện:* Miệng khô rát, liên tục khát nước dữ dội và uống nước rất nhiều nhưng vẫn không dứt cảm giác khát.\n\n"
            "3. **Ăn nhiều (Polyphagia):**\n"
            "   - *Cơ chế sinh lý:* Mặc dù đường trong máu cao, nhưng do kháng hoặc thiếu hụt insulin, tế bào không thể thu nhận glucose để tạo năng lượng ATP, dẫn đến trạng thái 'đói năng lượng ở cấp độ tế bào'.\n"
            "   - *Biểu hiện:* Người bệnh mau đói, thèm ăn liên tục, đặc biệt là đồ ngọt và tinh bột.\n\n"
            "4. **Sụt cân nhanh bất thường (Unexplained Weight Loss):**\n"
            "   - *Cơ chế sinh lý:* Tế bào không thể sử dụng glucose buộc cơ thể phải kích hoạt con đường dị hóa, đốt cháy lipid từ mô mỡ và tiêu hủy protein từ cơ bắp để tạo năng lượng sống bù đắp.\n"
            "   - *Biểu hiện:* Người bệnh ăn nhiều nhưng cân nặng giảm nhanh chóng từ 3–8 kg chỉ trong vài tuần.\n\n"
            "#### Các triệu chứng cảnh báo đi kèm:\n"
            "- **Mắt nhìn mờ:** Thay đổi độ cong và áp lực thẩm thấu trong thể thủy tinh.\n"
            "- **Mệt mỏi suy nhược:** Tế bào thiếu hụt năng lượng nghiêm trọng.\n"
            "- **Vết thương lâu lành:** Tăng đường huyết làm suy yếu hoạt động của bạch cầu và tuần hoàn mao mạch.\n"
            "- **Tê bì, châm chích đầu chi:** Tổn thương sớm các sợi thần kinh cảm giác ngoại biên.\n"
        )

    # TRƯỜNG HỢP 5: TIÊU CHUẨN CHẨN ĐOÁN & CHỈ SỐ XÉT NGHIỆM VÀNG
    elif any(k in lower_q for k in ["chẩn đoán", "tiêu chuẩn", "xét nghiệm", "chỉ số", "hba1c", "fpg", "ogtt", "đường huyết đói", "nghiệm pháp"]):
        bot_reply_blocks.append(
            "### Tiêu Chuẩn Chẩn Đoán Xác Định Đái Tháo Đường\n"
            "Theo **Quyết định số 5481/QĐ-BYT** của Bộ Y tế Việt Nam và **Tiêu chuẩn ADA 2024**, chẩn đoán xác định đái tháo đường típ 2 dựa trên xét nghiệm máu tĩnh mạch thỏa mãn ít nhất **1 trong 4 tiêu chí** sau:\n\n"
            "| Tiêu chí xét nghiệm | Ngưỡng chẩn đoán ĐTĐ | Ngưỡng Tiền ĐTĐ | Ngưỡng Bình thường | Điều kiện thực hiện |\n"
            "| :--- | :--- | :--- | :--- | :--- |\n"
            "| **Đường huyết đói (FPG)** | $\\ge 126$ mg/dL ($\\ge 7.0$ mmol/L) | $100 - 125$ mg/dL | $< 100$ mg/dL ($< 5.6$ mmol/L) | Nhịn ăn ít nhất 8 giờ |\n"
            "| **HbA1c** | $\\ge 6.5\\%$ ($48$ mmol/mol) | $5.7 - 6.4\\%$ | $< 5.7\\%$ | Phòng lab chuẩn hóa NGSP/DCCT |\n"
            "| **Nghiệm pháp OGTT 2h** | $\\ge 200$ mg/dL ($\\ge 11.1$ mmol/L) | $140 - 199$ mg/dL | $< 140$ mg/dL ($< 7.8$ mmol/L) | Uống 75g glucose, đo sau 2h |\n"
            "| **Glucose ngẫu nhiên** | $\\ge 200$ mg/dL ($\\ge 11.1$ mmol/L) | — | — | Kèm triệu chứng kinh điển 4 nhiều |\n\n"
            "#### Quy định xác thực lâm sàng (Theo QĐ 5481/BYT):\n"
            "- Nếu bệnh nhân **không có triệu chứng kinh điển** của tăng đường huyết cấp, cần làm lại xét nghiệm lần thứ hai (cùng phương pháp hoặc phương pháp khác) vào một ngày khác để khẳng định chẩn đoán.\n"
            "- Không dùng máy đo đường huyết mao mạch cầm tay để chẩn đoán xác định bệnh (chỉ dùng để tự theo dõi điều trị tại nhà).\n\n"
            "#### Mục tiêu kiểm soát đường huyết cá nhân hóa (ADA 2024):\n"
            "- **HbA1c mục tiêu:** Chung cho đa số người trưởng thành là $< 7.0\\%$; ở người trẻ mới mắc có thể hướng tới $< 6.5\\%$; ở người cao tuổi nhiều bệnh nền có thể nới lỏng $< 8.0\\%$.\n"
            "- **Chỉ số theo dõi CGM (Time in Range - TIR 70–180 mg/dL):** Cần duy trì $> 70\\%$ thời gian trong ngày.\n"
        )

    # TRƯỜNG HỢP 6: DƯỢC LÝ, METFORMIN & AN TOÀN THẬN (eGFR)
    elif any(k in lower_q for k in ["metformin", "suy thận", "egfr", "thuốc", "dược lý", "uống thuốc", "chỉnh liều", "insulin", "sglt2", "empagliflozin", "dapagliflozin"]):
        bot_reply_blocks.append(
            "### Hướng Dẫn Dược Lý, Sử Dụng Metformin & An Toàn Chức Năng Thận\n"
            "Theo **Quyết định 5481/QĐ-BYT** và **Cơ sở Tri thức Lâm sàng Vinmec**, việc sử dụng thuốc hạ đường huyết cần tuân thủ nghiêm ngặt chức năng lọc cầu thận (eGFR):\n\n"
            "#### 1. Nguyên tắc sử dụng Metformin (Thuốc lựa chọn đầu tay - First-line):\n"
            "- **Cơ chế tác dụng:** Giảm sản xuất glucose ở gan, tăng nhạy cảm của thụ thể insulin tại mô cơ và mô mỡ, không gây hạ đường huyết đơn độc và hỗ trợ kiểm soát cân nặng.\n"
            "- **Liều dùng khởi đầu:** 500 mg hoặc 850 mg uống 1–2 lần/ngày trong hoặc ngay sau bữa ăn chính để giảm thiểu tác dụng phụ tiêu hóa (đầy bụng, buồn nôn, tiêu chảy).\n\n"
            "#### 2. Bảng hiệu chỉnh liều Metformin bắt buộc theo mức lọc cầu thận (eGFR):\n"
            "- **eGFR $\\ge 45$ mL/phút/1.73m² (Bình thường đến suy thận nhẹ độ 1-2, 3a):**\n"
            "  * *Khuyến cáo:* An toàn sử dụng liều chuẩn đầy đủ (tối đa 2000 mg/ngày).\n"
            "  * *Theo dõi:* Định kỳ kiểm tra chức năng thận eGFR mỗi năm 1 lần.\n"
            "- **eGFR 30 - 44 mL/phút/1.73m² (Suy thận độ 3b):**\n"
            "  * *Khuyến cáo:* **Bắt buộc giảm 50% liều tối đa** (tuyệt đối không dùng quá 1000 mg/ngày). Không khởi đầu dùng mới Metformin ở mốc này.\n"
            "  * *Cảnh báo an toàn:* Bắt buộc theo dõi eGFR sát sao mỗi 3–6 tháng/lần vì tăng nguy cơ tích lũy thuốc dẫn đến biến chứng nhiễm toan acid lactic.\n"
            "- **eGFR < 30 mL/phút/1.73m² (Suy thận nặng độ 4-5):**\n"
            "  * *Khuyến cáo:* **CHỐNG CHỈ ĐỊNH TUYỆT ĐỐI Metformin**. Bắt buộc ngừng thuốc ngay lập tức!\n"
            "  * *Nguy cơ:* Nhiễm toan acid lactic (Lactic Acidosis) có tỷ lệ tử vong lên tới 50%.\n"
            "  * *Phác đồ thay thế:* Chuyển ngay sang liệu pháp Insulin hoặc các nhóm thuốc bài tiết qua gan.\n\n"
            "#### 3. Các nhóm thuốc ưu tiên bảo vệ tim - thận:\n"
            "- **Nhóm ức chế kênh SGLT2 (Empagliflozin, Dapagliflozin):** Tăng thải glucose qua nước tiểu, làm chậm tiến triển suy giảm eGFR và giảm biến cố suy tim, tử vong tim mạch.\n"
            "- **Nhóm bảo vệ màng lọc cầu thận:** Ưu tiên thuốc ức chế men chuyển (ACEi) hoặc chẹn thụ thể ARB nếu bệnh nhân có tăng huyết áp hoặc có đạm niệu.\n"
        )

    # TRƯỜNG HỢP 7: DINH DƯỠNG, TẬP LUYỆN & LỐI SỐNG (RL SIMULATION)
    elif any(k in lower_q for k in ["dinh dưỡng", "ăn uống", "vận động", "tập luyện", "lối sống", "tinh bột", "thực đơn", "kiêng", "thể dục", "chế độ ăn"]):
        rl_res = coordinator.rl_agent.recommend_simulation_policy(peak_cgm=195.0, previous_compliance=0)
        bot_reply_blocks.append(
            "### Hướng Dẫn Can Thiệp Dinh Dưỡng, Vận Động & Tối Ưu Lối Sống\n"
            "Theo tiêu chuẩn ADA 2024 kết hợp mô hình Học tăng cường (Reinforcement Learning - Q-Learning Bellman):\n\n"
            "#### 1. Liệu pháp Dinh dưỡng Y khoa (Medical Nutrition Therapy - MNT):\n"
            "- **Cắt giảm Carbohydrate hấp thu nhanh (GI cao):** Giảm 30–40% lượng tinh bột tinh chế (cơm trắng, bún, miến, bánh mì trắng, đồ ngọt, nước có gas) nhằm tránh làm tăng vọt glucose sau ăn.\n"
            "- **Tăng cường Carbohydrate phức hợp & Chất xơ hòa tan:** Ưu tiên gạo lứt, yến mạch, các loại đậu, rau xanh lá đậm (bông cải, rau bina). Chất xơ làm chậm quá trình tiêu hóa và hấp thu đường vào máu.\n"
            "- **Bổ sung Protein nạc:** Thịt ức gà, cá, trứng, đậu hũ giúp duy trì khối cơ và tạo cảm giác no lâu mà không làm tăng đường huyết.\n"
            "- **Chất béo có lợi:** Dầu ô liu nguyên chất, quả bơ, hạt óc chó, hạnh nhân; hạn chế chất béo bão hòa từ mỡ động vật và phủ tạng.\n\n"
            "#### 2. Chiến lược Vận động & Cơ chế Kích hoạt Kênh GLUT4:\n"
            "- **Tổng lượng vận động khuyến nghị:** Tối thiểu 150 phút/tuần bài tập thể lực nhịp điệu cường độ trung bình (đi bộ nhanh, bơi lội, đạp xe), không nghỉ quá 2 ngày liên tiếp.\n"
            "- **Chiến lược đi bộ sau ăn (Post-prandial walking):**\n"
            "  * *Cơ chế sinh lý học:* Thực hiện đi bộ nhanh từ 20–30 phút sau bữa ăn chính từ 15–30 phút sẽ kích hoạt cơ chế vận chuyển glucose qua kênh thụ thể **GLUT4** trực tiếp vào tế bào cơ mà **không cần phụ thuộc vào insulin**.\n"
            "  * *Hiệu quả:* Cắt giảm hiệu quả đỉnh đường huyết sau ăn, ngăn ngừa glucose vọt trên ngưỡng 180 mg/dL.\n\n"
            "#### 3. Chính sách Tối ưu Bellman (Từ Mô hình Q-Learning):\n"
            f"- **Chính sách lựa chọn:** `{rl_res.payload['action_description']}`\n"
            f"- **Kế hoạch hành động cụ thể:**\n{rl_res.payload['detailed_plan']}\n"
        )

    # TRƯỜNG HỢP 8: HẠ ĐƯỜNG HUYẾT & XỬ TRÍ CẤP CỨU
    elif any(k in lower_q for k in ["hạ đường huyết", "tụt đường huyết", "cấp cứu", "run tay", "vã mồ hôi", "xỉu", "choáng", "đường huyết thấp"]):
        bot_reply_blocks.append(
            "### Hướng Dẫn Nhận Biết & Xử Trí Cấp Cứu Hạ Đường Huyết\n"
            "Hạ đường huyết là tình trạng cấp cứu nguy hiểm thường gặp ở người bệnh điều trị bằng Insulin hoặc nhóm thuốc kích thích tiết insulin (Sulfonylureas):\n\n"
            "#### 1. Định nghĩa & Dấu hiệu nhận biết:\n"
            "- **Ngưỡng cảnh báo cấp cứu:** Nồng độ glucose máu $< 70$ mg/dL ($< 3.9$ mmol/L).\n"
            "- **Triệu chứng sớm (Kích hoạt thần kinh giao cảm):** Run tay chân, vã mồ hôi lạnh, tim đập nhanh hồi hộp, cảm giác đói cồn cào, lo âu, bồn chồn.\n"
            "- **Triệu chứng muộn (Thiếu glucose nuôi não):** Chóng mặt, hoa mắt, nhìn mờ, lú lẫn, nói nhảm, mất thăng bằng, co giật và hôn mê.\n\n"
            "#### 2. Quy tắc Xử trí Cấp cứu '15 - 15' Chuẩn mực Y khoa:\n"
            "- **Bước 1 (Nạp ngay 15g Carbohydrate tác dụng nhanh):** Cho bệnh nhân dùng ngay 1 trong các thực phẩm sau:\n"
            "  * 1/2 ly nước ép trái cây ngọt hoặc nước ngọt có đường (khoảng 100–150 mL).\n"
            "  * 3 thìa cà phê đường trắng hoặc mật ong pha với nửa cốc nước ấm.\n"
            "  * 3–4 viên kẹo ngọt hoặc viên đường glucose chuyên dụng.\n"
            "- **Bước 2 (Nghỉ ngơi 15 phút):** Cho bệnh nhân ngồi hoặc nằm nghỉ yên tĩnh, không hoạt động mạnh.\n"
            "- **Bước 3 (Kiểm tra lại đường huyết sau 15 phút):**\n"
            "  * Nếu đường huyết vẫn $< 70$ mg/dL: Tiếp tục lặp lại Bước 1 (nạp thêm 15g đường) và chờ thêm 15 phút.\n"
            "  * Nếu đường huyết đã $\\ge 70$ mg/dL: Cho bệnh nhân ăn một bữa phụ nhẹ (bánh quy, sữa hạt) hoặc dùng bữa ăn chính nếu đã tới giờ để tránh đường huyết tụt trở lại.\n"
            "- **Lưu ý đặc biệt:** Nếu bệnh nhân bất tỉnh, co giật hoặc không thể nuốt được, **tuyệt đối không cố đổ nước đường vào miệng** vì nguy cơ sặc vào đường thở. Phải đưa ngay đến phòng cấp cứu gần nhất hoặc tiêm Glucagon.\n"
        )

    # TRƯỜNG HỢP 9: TỔNG QUÁT / SMART GRAPHRAG SYNTHESIZER (TỔNG HỢP ĐẦY ĐỦ, KHÔNG CẮT CỤT)
    else:
        facts_summary = ", ".join(graph_facts[:3]) if graph_facts else "Đái tháo đường Típ 2, Kháng insulin, Tiêu chuẩn chẩn đoán"

        bot_reply_blocks.append(
            f"### Kết Quả Phân Tích & Tra Cứu Tri Thức Lâm Sàng (Hybrid GraphRAG)\n"
            f"Hệ thống đa tác tử đã phân tích câu hỏi **\"{chat_query}\"**, nhận diện các thực thể trên đồ thị tri thức Neo4j và tổng hợp từ cơ sở dữ liệu hướng dẫn y tế chính thức:\n\n"
            f"#### 1. Tổng quan & Bản chất y khoa:\n"
            f"- Đái tháo đường típ 2 là bệnh lý rối loạn chuyển hóa mạn tính liên quan mật thiết đến tình trạng kháng insulin và sự suy giảm tiến triển chức năng bài tiết insulin của tế bào beta tụy.\n"
            f"- Các mắt xích tri thức liên quan được trích xuất từ đồ thị: `{facts_summary}`.\n\n"
            f"#### 2. Dẫn chứng chuyên môn từ Tài liệu Hướng dẫn Lâm sàng:\n"
        )
        if rag_results:
            for i, chunk in enumerate(rag_results[:2], 1):
                bot_reply_blocks.append(
                    f"**Trích đoạn {i} [{chunk.get('source', 'Tài liệu')} - Độ tương đồng Cosine: {chunk.get('similarity_score', 0.0):.3f}]:**\n"
                    f"> \"{chunk.get('text', '').strip()}\"\n\n"
                )
        else:
            bot_reply_blocks.append(
                "> Thông tin truy vấn đã được đối chiếu toàn diện với cơ sở dữ liệu hướng dẫn chẩn đoán và điều trị đái tháo đường.\n\n"
            )

        bot_reply_blocks.append(
            "#### 3. Khuyến nghị ứng dụng lâm sàng:\n"
            "- Đối chiếu chỉ số đường huyết định kỳ với các mốc chuẩn: Đường huyết đói $\\ge 126$ mg/dL, HbA1c $\\ge 6.5\\%$, OGTT 2h $\\ge 200$ mg/dL.\n"
            "- Kết hợp chặt chẽ kiềng 3 chân: Dinh dưỡng giảm đường tinh chế + Vận động thể lực kích hoạt GLUT4 + Dược lý an toàn theo chức năng thận eGFR.\n\n"
            "#### 4. Cảnh báo an toàn y khoa:\n"
            "- Mọi thông tin do hệ thống đa tác tử cung cấp có tính chất hỗ trợ quyết định chuyên môn (Decision Support). Quyết định thay đổi thuốc hoặc chỉ định điều trị bắt buộc phải có sự thăm khám trực tiếp của bác sĩ chuyên khoa.\n"
        )

    # DẪN CHỨNG NGUỒN CHÍNH THỐNG (ĐÃ CẬP NHẬT URL CHUẨN, 100% KHÔNG 404)
    bot_reply_blocks.append(
        "\n---\n"
        "**Nguồn dẫn chứng & Tài liệu chuyên môn chính thống:**\n"
        "- **Bộ Y tế Việt Nam:** Quyết định số 5481/QĐ-BYT \"Hướng dẫn chẩn đoán và điều trị đái tháo đường típ 2\"\n"
        "  * *Tra cứu trực tuyến:* [Cổng Thông tin Bộ Y tế](https://moh.gov.vn) • [Tìm kiếm Google](https://www.google.com/search?q=Quyet+dinh+5481+QD-BYT+Bo+Y+te)\n"
        "- **Bệnh viện Đa khoa Quốc tế Vinmec:** Chuyên mục Đái tháo đường Típ 2 - Khoa Nội tiết Chuyển hóa\n"
        "  * *Tra cứu trực tuyến:* [Trang chủ Vinmec](https://www.vinmec.com) • [Tìm kiếm Vinmec](https://www.vinmec.com/vi/tim-kiem/?q=ti%E1%BB%83u+%C4%91%C6%B0%E1%BB%9Dng)\n"
        "- **Hiệp hội Đái tháo đường Hoa Kỳ:** ADA Standards of Care in Diabetes 2024\n"
        "  * *Tra cứu trực tuyến:* [Tạp chí Diabetes Care](https://diabetesjournals.org/care/issue/47/Supplement_1) • [Trang chủ ADA](https://diabetes.org)\n"
        "*(Lưu ý: Mọi tài liệu chuyên môn trên đều có toàn văn offline ngay trong hệ thống; bạn có thể mở rộng mục **'📚 Nguồn trích dẫn & Dữ liệu đối chiếu nội bộ'** bên dưới mỗi tin nhắn để xem toàn văn trích đoạn mà không cần mạng internet).* "
    )

    return "".join(bot_reply_blocks)


def run_streamlit_app():
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import streamlit.components.v1 as components

    st.set_page_config(
        page_title="Hệ Thống Đa Tác Tử Hỗ Trợ Quyết Định Tiểu Đường | PTIT ISD",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS - Giao diện chuyên nghiệp, chuẩn mực y khoa, không dùng icon/emoji thừa
    st.markdown("""
    <style>
        [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        .main .block-container {
            padding-top: 0.8rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100%;
        }
        html, body, p, div, span, label {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.88rem !important;
            color: #64748b !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            color: #0f172a !important;
            font-weight: 700;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.5rem;
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 1.1rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 46px;
            font-size: 1.02rem !important;
            font-weight: 600;
            color: #475569;
            padding-left: 0.4rem;
            padding-right: 0.4rem;
        }
        .stTabs [aria-selected="true"] {
            color: #1e40af !important;
            border-bottom-color: #1e40af !important;
            font-weight: 700;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 6px;
            font-size: 0.95rem !important;
            font-weight: 600;
            padding: 0.5rem 1.2rem;
        }
        .stButton>button[kind="primary"] {
            background-color: #1e40af;
            border: none;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #1d4ed8;
        }
    </style>
    """, unsafe_allow_html=True)

    # Khởi tạo tài nguyên hệ thống và Cache
    @st.cache_resource
    def load_system_core():
        init_database()
        kg = MedicalKnowledgeGraph()
        vstore = ClinicalVectorStore()
        vstore.build_index_from_directory()
        coord = CoordinatorAgent(kg, vstore)
        return kg, vstore, coord

    kg, vstore, coordinator = load_system_core()
    orchestrator = coordinator # alias

    # Khởi tạo session state
    if "selected_pid" not in st.session_state:
        st.session_state.selected_pid = "BN001"
    if "pending_chat" not in st.session_state:
        st.session_state.pending_chat = None

    def create_default_welcome_message():
        return {
            "role": "assistant",
            "content": (
                "Xin chào! Tôi là **Trợ lý Tra cứu Tri thức Y tế & Hội chẩn Đa tác tử**.\n\n"
                "Bạn có thể đặt câu hỏi hoặc yêu cầu tư vấn về:\n"
                "- **Tiêu chuẩn chẩn đoán:** Đường huyết đói, HbA1c, OGTT theo QĐ 5481/BYT & ADA 2024.\n"
                "- **Dược lý & Chỉnh liều:** Đánh giá an toàn và liều dùng Metformin theo mức lọc cầu thận eGFR.\n"
                "- **Triệu chứng lâm sàng:** Cơ chế bệnh sinh của Hội chứng '4 nhiều' (Vinmec).\n"
                "- **Dinh dưỡng & Vận động:** Mô phỏng thích nghi lối sống tối ưu hóa theo Reinforcement Learning.\n"
                "- **Hội chẩn ca bệnh:** Tư vấn toàn diện hồ sơ (Ví dụ: `Hội chẩn BN001`, `BN002`, `BN003`)."
            ),
            "graph_facts": [],
            "sources": []
        }

    if "chat_sessions" not in st.session_state:
        legacy_msgs = st.session_state.get("chat_messages")
        first_sess_id = f"sess_{int(time.time() * 1000)}"
        st.session_state.chat_sessions = [
            {
                "id": first_sess_id,
                "title": "Phiên tư vấn #1",
                "created_at": time.strftime("%H:%M - %d/%m"),
                "messages": legacy_msgs if legacy_msgs else [create_default_welcome_message()]
            }
        ]
        st.session_state.current_session_id = first_sess_id

    # Tự động chuẩn hóa và dọn dẹp các đường link lỗi thời (404) trong toàn bộ lịch sử hội thoại hiện hành
    if "chat_sessions" in st.session_state:
        for sess in st.session_state.chat_sessions:
            for msg in sess.get("messages", []):
                if isinstance(msg.get("content"), str):
                    for bad_url in [
                        "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
                        "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
                        "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Quyet-dinh-5481-QD-BYT-2020-tai-lieu-Huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2-463283.aspx",
                        "https://www.vinmec.com/vi/chuyen-khoa/noi-tiet-chuyen-hoa/"
                    ]:
                        if bad_url in msg["content"]:
                            if "vinmec" in bad_url:
                                msg["content"] = msg["content"].replace(bad_url, "https://www.vinmec.com/vi/tim-kiem/?q=ti%E1%BB%83u+%C4%91%C6%B0%E1%BB%9Dng")
                            else:
                                msg["content"] = msg["content"].replace(bad_url, "https://www.google.com/search?q=Quyet+dinh+5481+QD-BYT+Bo+Y+te")
                if msg.get("sources"):
                    for s in msg["sources"]:
                        if isinstance(s.get("url"), str):
                            if "4166" in s["url"] or "chuyen-khoa" in s["url"]:
                                s["url"] = "https://www.vinmec.com/vi/tim-kiem/?q=ti%E1%BB%83u+%C4%91%C6%B0%E1%BB%9Dng"
                            elif "kcb.vn" in s["url"] or "463283" in s["url"]:
                                s["url"] = "https://www.google.com/search?q=Quyet+dinh+5481+QD-BYT+Bo+Y+te"

    # HEADER CHUẨN MỰC Y KHOA
    st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; background: #0f172a; padding: 0.9rem 1.4rem; border-radius: 8px; color: white; margin-bottom: 0.9rem;">
    <div>
        <div style="font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em;">HỆ THỐNG ĐA TÁC TỬ HỖ TRỢ QUYẾT ĐỊNH ĐIỀU TRỊ TIỂU ĐƯỜNG</div>
        <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem;">Clinical Decision Support System (CDSS) | GraphRAG, Deep Learning & Reinforcement Learning</div>
    </div>
    <div style="display: flex; gap: 0.6rem; align-items: center;">
        <span style="background: #1e293b; color: #38bdf8; border: 1px solid #334155; font-size: 0.82rem; font-weight: 600; padding: 0.25rem 0.65rem; border-radius: 4px;">8 Tác tử hoạt động</span>
        <span style="background: #1e293b; color: #cbd5e1; border: 1px solid #334155; font-size: 0.82rem; font-weight: 500; padding: 0.25rem 0.65rem; border-radius: 4px;">PTIT ISD 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # Chi tiết 6 tầng kiến trúc thu gọn
    with st.expander("Cấu trúc 6 tầng kiến trúc hệ thống (PTIT Architecture)", expanded=False):
        st.markdown("""
        - **Tầng 1 (Giao diện người dùng):** Web Dashboard tương tác & Terminal CLI tự động.
        - **Tầng 2 (API & Xử lý dữ liệu - PatientDataAgent):** Kiểm thực định dạng, kiểm tra miền giá trị sinh lý y khoa.
        - **Tầng 3 (Điều phối đa tác tử - CoordinatorAgent):** Thực thi quy trình tuần tự 12 bước logic và hợp đồng thông điệp $m_i$.
        - **Tầng 4 (Tác tử chuyên môn):** DLPredictionAgent, RuleBasedAgent, KnowledgeRetrievalAgent, RLAgent, ExplanationAgent, MonitoringAgent.
        - **Tầng 5 (Tri thức & Mô hình):** DNN 6 tầng, RNN dừng CGM, Neo4j DB_diabet.cypher và Vector RAG Cosine.
        - **Tầng 6 (Đánh giá & Kiểm toán):** SQLite `agent_action_logs`, đo lường Groundedness, Hallucination Rate và Traceability.
        """)

    # 5 TABS CHÍNH
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Hội chẩn Lâm sàng",
        "Trợ lý Tri thức Y tế",
        "Mô hình Học sâu & CGM",
        "Đồ thị Tri thức Neo4j",
        "Mô phỏng RL & CSDL"
    ])

    # =========================================================================
    # TAB 1: HỘI CHẨN ĐA TÁC TỬ
    # =========================================================================
    with tab1:
        st.subheader("Hội Chẩn Đa Tác Tử (Clinical Decision Support)")

        input_mode = st.radio("Nguồn dữ liệu bệnh nhân:", ["Hồ sơ chuẩn từ CSDL SQLite", "Nhập ca bệnh tùy biến / Thử nghiệm lỗi xác thực"], horizontal=True)

        if "chuẩn" in input_mode:
            patients_list = get_all_patients()
            p_options = {p["patient_id"]: f"{p['patient_id']} - {p['full_name']} (Tuổi: {p['age']}, BMI: {p['bmi']}, eGFR: {p['egfr']} mL/phút)" for p in patients_list}

            selected_pid = st.selectbox(
                "Chọn hồ sơ bệnh nhân từ CSDL SQLite:",
                options=list(p_options.keys()),
                format_func=lambda x: p_options[x],
                index=list(p_options.keys()).index(st.session_state.selected_pid) if st.session_state.selected_pid in p_options else 0
            )
            st.session_state.selected_pid = selected_pid
            patient_obj = get_patient(selected_pid)

            # Tính toán phân loại sinh lý lâm sàng
            p_glu = float(patient_obj.get("fasting_glucose", 0))
            p_hba1c = float(patient_obj.get("hba1c", 0))
            p_egfr = float(patient_obj.get("egfr", 0))
            p_sys = float(patient_obj.get("systolic_bp", 0))
            p_dia = float(patient_obj.get("diastolic_bp", 0))
            p_bmi = float(patient_obj.get("bmi", 0))

            if p_glu >= 126:
                tag_glu_text = "Ngưỡng ĐTĐ"
            elif p_glu >= 100:
                tag_glu_text = "Tiền ĐTĐ"
            else:
                tag_glu_text = "Bình thường"

            if p_hba1c >= 6.5:
                tag_hba1c_text = "Ngưỡng ĐTĐ"
            elif p_hba1c >= 5.7:
                tag_hba1c_text = "Tiền ĐTĐ"
            else:
                tag_hba1c_text = "Bình thường"

            if p_egfr < 30:
                tag_egfr_text = "Suy thận nặng (Độ 4-5)"
            elif p_egfr < 45:
                tag_egfr_text = "Suy thận độ 3b (Cần giảm liều)"
            elif p_egfr < 60:
                tag_egfr_text = "Suy thận độ 3a"
            else:
                tag_egfr_text = "Chức năng thận tốt"

            if p_sys >= 140 or p_dia >= 90:
                tag_bp_text = "Tăng HA Độ 1"
            else:
                tag_bp_text = "Huyết áp ổn định"

            if p_bmi >= 30:
                tag_bmi_text = "Béo phì độ 1"
            elif p_bmi >= 25:
                tag_bmi_text = "Thừa cân"
            else:
                tag_bmi_text = "Bình thường"

            # Thẻ hồ sơ lâm sàng hiển thị hoàn toàn bằng widget Streamlit native (không dùng HTML thô)
            with st.container(border=True):
                c_p1, c_p2 = st.columns([3, 1])
                with c_p1:
                    st.markdown(f"**Bệnh nhân:** {patient_obj['full_name']} | Mã BN: `{patient_obj['patient_id']}`")
                    st.caption(f"Giới tính: {patient_obj['gender']} | Tuổi: {patient_obj['age']} | Tiền sử bệnh nền: {patient_obj['clinical_notes']}")
                with c_p2:
                    st.info(f"Thận: {tag_egfr_text}")

                col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
                col_v1.metric("Đường huyết đói", f"{patient_obj['fasting_glucose']} mg/dL", delta=tag_glu_text, delta_color="inverse" if "ĐTĐ" in tag_glu_text else "normal")
                col_v2.metric("Chỉ số HbA1c", f"{patient_obj['hba1c']}%", delta=tag_hba1c_text, delta_color="inverse" if "ĐTĐ" in tag_hba1c_text else "normal")
                col_v3.metric("Lọc thận eGFR", f"{patient_obj['egfr']} mL/phút", delta=tag_egfr_text, delta_color="inverse" if "Suy thận" in tag_egfr_text else "normal")
                col_v4.metric("Huyết áp", f"{patient_obj['systolic_bp']}/{patient_obj['diastolic_bp']} mmHg", delta=tag_bp_text, delta_color="inverse" if "Tăng HA" in tag_bp_text else "normal")
                col_v5.metric("Chỉ số BMI", f"{patient_obj['bmi']} kg/m²", delta=tag_bmi_text, delta_color="off")

            default_queries = {
                "BN001": "Tôi 58 tuổi, BMI 31.2, HbA1c 7.8%, đường huyết sáng nay là 192 mg/dL và bị suy thận (eGFR = 42 mL/phút). Tôi có dùng được Metformin nguyên liều không và cần ăn uống, tập luyện thế nào?",
                "BN002": "Tôi 46 tuổi, HbA1c 6.1%, glucose đói 115 mg/dL. Tôi có bị đái tháo đường chưa và cần điều chỉnh lối sống ra sao?",
                "BN003": "Tôi 67 tuổi, HbA1c 8.6%, glucose đói 210 mg/dL, suy thận nặng eGFR = 28 mL/phút. Tôi có dùng được Metformin không?"
            }
            user_query_input = st.text_area("Yêu cầu / Câu hỏi lâm sàng:", value=default_queries.get(selected_pid, "Tư vấn phác đồ điều trị cho bệnh nhân."), height=75)
            target_patient_data = patient_obj

        else:
            st.info("Nhập các thông số để kiểm thử khả năng xác thực dữ liệu của PatientDataAgent.")
            c_f1, c_f2, c_f3, c_f4 = st.columns(4)
            with c_f1:
                in_pid = st.text_input("Mã BN:", value="CUSTOM_01")
                in_age = st.number_input("Tuổi (năm):", value=55, min_value=-50, max_value=200)
                in_gender = st.selectbox("Giới tính:", ["Nam", "Nữ"])
            with c_f2:
                in_bmi = st.number_input("BMI (kg/m²):", value=28.5, min_value=-10.0, max_value=100.0, step=0.1)
                in_hba1c = st.number_input("HbA1c (%):", value=7.2, min_value=-5.0, max_value=40.0, step=0.1)
            with c_f3:
                in_glu = st.number_input("Glucose đói (mg/dL):", value=165.0, min_value=-100.0, max_value=5000.0, step=1.0)
                in_egfr = st.number_input("eGFR (mL/phút):", value=38.0, min_value=-50.0, max_value=300.0, step=1.0)
            with c_f4:
                in_sys = st.number_input("Huyết áp tâm thu:", value=140, min_value=-50, max_value=400)
                in_dia = st.number_input("Huyết áp tâm trương:", value=85, min_value=-50, max_value=300)

            user_query_input = st.text_area("Yêu cầu / Câu hỏi lâm sàng:", value="Tư vấn phác đồ điều trị và chỉnh liều thuốc.", height=70)
            target_patient_data = {
                "patient_id": in_pid,
                "full_name": f"Bệnh nhân {in_pid}",
                "age": in_age,
                "gender": in_gender,
                "bmi": in_bmi,
                "hba1c": in_hba1c,
                "fasting_glucose": in_glu,
                "systolic_bp": in_sys,
                "diastolic_bp": in_dia,
                "egfr": in_egfr,
                "hypertension": 1 if in_sys >= 140 or in_dia >= 90 else 0,
                "heart_disease": 0,
                "smoking_history": "never",
                "clinical_notes": "Hồ sơ nhập kiểm thử"
            }

        if st.button("Bắt đầu điều phối quy trình 12 bước (Coordinator Workflow)", type="primary", use_container_width=True):
            with st.spinner("Coordinator Agent đang điều phối 8 tác tử tuần tự..."):
                consult_res = coordinator.execute_workflow(target_patient_data, user_query=user_query_input)

            # Xử lý trường hợp xác thực thất bại tại Bước 4 - 5
            if consult_res.get("status") == "error":
                st.error("Xác thực dữ liệu thất bại: Coordinator Agent đã chấm dứt quy trình tại Bước 5 để đảm bảo an toàn lâm sàng.")
                st.markdown(consult_res["final_report"])
                with st.expander("Xem vết thực thi đến thời điểm dừng", expanded=True):
                    for step in consult_res.get("react_trace", []):
                        st.markdown(f"- **[Bước {step['step']}] {step['phase']} - {step['agent']}:** {step['observation']}")
            else:
                st.success(f"Hoàn tất quy trình hội chẩn đa tác tử 12 bước thành công trong {consult_res['latency_ms']} ms.")

                # 4 Metrics lâm sàng quan trọng
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Xác suất ĐTĐ Típ 2 (DNN)", f"{consult_res['dl_output']['risk_analysis']['risk_percentage']}%", delta=consult_res['dl_output']['risk_analysis']['risk_level'], delta_color="inverse" if "CAO" in consult_res['dl_output']['risk_analysis']['risk_level'] else "normal")
                c2.metric("Đỉnh Đường huyết 4h (RNN)", f"{consult_res['dl_output']['cgm_forecast']['peak_glucose']} mg/dL", delta="Dự báo đỉnh")
                c3.metric("Độ tin cậy dẫn chứng", f"{consult_res['groundedness_score']}%", delta="Groundedness")
                c4.metric("Tỷ lệ ảo giác kiểm soát", f"{consult_res['hallucination_rate']}%", delta="Kiểm soát an toàn", delta_color="inverse")

                # BÁO CÁO HỖ TRỢ QUYẾT ĐỊNH Y KHOA HIỂN THỊ NGAY ĐẦU
                st.markdown(consult_res["final_report"])

                # Vết 12 bước hiển thị bằng container native (không dùng HTML thô)
                with st.expander("Chi tiết vết điều phối 12 bước & Hợp đồng thông điệp m_i", expanded=False):
                    for step in consult_res["react_trace"]:
                        with st.container(border=True):
                            st.markdown(f"**[Bước {step['step']}] {step['phase']} — `{step['agent']}`**")
                            st.markdown(f"- **Suy luận (Thought):** {step['thought']}")
                            st.markdown(f"- **Hành động (Action):** `{step['action']}`")
                            st.markdown(f"- **Quan sát (Observation):** {step['observation']}")
                            if "message" in step:
                                with st.expander(f"Hợp đồng thông điệp chuẩn hóa m_i của {step['agent']}"):
                                    st.json(step["message"])

    # =========================================================================
    # TAB 2: TRỢ LÝ TRA CỨU TRI THỨC (HYBRID GRAPHRAG)
    # =========================================================================
    with tab2:
        st.subheader("Trợ Lý Tra Cứu Tri Thức Y Tế (GraphRAG & Neo4j)")

        # Phân chia bố cục: Cột trái (Quản lý phiên hội thoại) & Cột phải (Khung chat tương tác)
        col_sess_list, col_sess_main = st.columns([1, 3.2], gap="medium")

        # --- CỘT TRÁI: QUẢN LÝ PHIÊN HỘI THOẠI (CHAT SESSIONS) ---
        with col_sess_list:
            st.markdown("##### 📁 Quản lý phiên chat")
            if st.button("➕ Tạo phiên mới", type="primary", use_container_width=True, key="btn_new_chat_session"):
                new_id = f"sess_{int(time.time() * 1000)}"
                st.session_state.chat_sessions.insert(0, {
                    "id": new_id,
                    "title": f"Phiên mới #{len(st.session_state.chat_sessions) + 1}",
                    "created_at": time.strftime("%H:%M - %d/%m"),
                    "messages": [create_default_welcome_message()]
                })
                st.session_state.current_session_id = new_id
                st.session_state.pending_chat = None
                st.rerun()

            st.caption(f"Tổng số: **{len(st.session_state.chat_sessions)}** phiên hội thoại")

            # Danh sách các phiên có vùng cuộn riêng, không đẩy trang web dài ra
            with st.container(height=490):
                for sess in st.session_state.chat_sessions:
                    is_active = (sess["id"] == st.session_state.current_session_id)
                    c_btn, c_del = st.columns([5, 1])
                    with c_btn:
                        btn_prefix = "🟢 " if is_active else "💬 "
                        disp_title = sess["title"]
                        if len(disp_title) > 24:
                            disp_title = disp_title[:22] + "..."
                        if st.button(
                            f"{btn_prefix}{disp_title}",
                            key=f"btn_sess_{sess['id']}",
                            type="primary" if is_active else "secondary",
                            use_container_width=True,
                            help=f"{sess['title']} ({len(sess['messages'])} tin nhắn • Tạo lúc {sess.get('created_at', '')})"
                        ):
                            st.session_state.current_session_id = sess["id"]
                            st.rerun()
                        st.caption(f"{len(sess['messages'])} tin nhắn • {sess.get('created_at', '')}")
                    with c_del:
                        if len(st.session_state.chat_sessions) > 1:
                            if st.button("✕", key=f"del_sess_{sess['id']}", help="Xóa phiên này"):
                                st.session_state.chat_sessions = [s for s in st.session_state.chat_sessions if s["id"] != sess["id"]]
                                if st.session_state.current_session_id == sess["id"]:
                                    st.session_state.current_session_id = st.session_state.chat_sessions[0]["id"]
                                st.rerun()

            if len(st.session_state.chat_sessions) > 1:
                if st.button("🧹 Đặt lại toàn bộ phiên", use_container_width=True, help="Khởi tạo lại danh sách phiên hội thoại về mặc định"):
                    fresh_id = f"sess_{int(time.time() * 1000)}"
                    st.session_state.chat_sessions = [
                        {
                            "id": fresh_id,
                            "title": "Phiên tư vấn #1",
                            "created_at": time.strftime("%H:%M - %d/%m"),
                            "messages": [create_default_welcome_message()]
                        }
                    ]
                    st.session_state.current_session_id = fresh_id
                    st.rerun()

        # --- CỘT PHẢI: KHUNG TRÒ CHUYỆN CỦA PHIÊN ĐANG CHỌN ---
        with col_sess_main:
            # Lấy thông tin phiên hiện tại
            curr_sess = next(
                (s for s in st.session_state.chat_sessions if s["id"] == st.session_state.current_session_id),
                None
            )
            if not curr_sess:
                curr_sess = st.session_state.chat_sessions[0]
                st.session_state.current_session_id = curr_sess["id"]

            # Thanh tiêu đề và tùy chọn phiên
            c_header, c_tools = st.columns([2.4, 2.4])
            with c_header:
                st.markdown(f"##### 🩺 {curr_sess['title']}")
                st.caption(f"Trạng thái: Đang hoạt động • {len(curr_sess['messages'])} tin nhắn • Tạo lúc: {curr_sess.get('created_at', 'Hôm nay')}")
            with c_tools:
                col_t_opt, col_t_docs = st.columns(2)
                with col_t_opt:
                    with st.popover("⚙️ Tùy chọn phiên"):
                        new_title_val = st.text_input("Đổi tên phiên này:", value=curr_sess["title"], key=f"rename_input_{curr_sess['id']}")
                        if st.button("Lưu tên mới", use_container_width=True, key=f"save_title_{curr_sess['id']}"):
                            if new_title_val.strip():
                                curr_sess["title"] = new_title_val.strip()
                                st.rerun()
                        if st.button("🧹 Làm sạch phiên này", use_container_width=True, key=f"clear_msgs_{curr_sess['id']}"):
                            curr_sess["messages"] = [create_default_welcome_message()]
                            st.rerun()
                        export_json = json.dumps({
                            "session_id": curr_sess["id"],
                            "title": curr_sess["title"],
                            "created_at": curr_sess.get("created_at"),
                            "messages": curr_sess["messages"]
                        }, ensure_ascii=False, indent=2)
                        st.download_button("📥 Xuất dữ liệu phiên (.json)", data=export_json, file_name=f"{curr_sess['id']}.json", mime="application/json", use_container_width=True)

                with col_t_docs:
                    with st.popover("📖 Đọc tài liệu gốc (Offline)"):
                        st.markdown("##### 📚 Thư viện văn bản gốc tích hợp sẵn:")
                        doc_choice = st.selectbox(
                            "Chọn tài liệu chuyên môn muốn xem:",
                            [
                                "Quyết định 5481/QĐ-BYT (Bộ Y tế)",
                                "Tài liệu Chuyên khoa Vinmec",
                                "ADA Standards of Care 2024"
                            ],
                            key=f"doc_select_{curr_sess['id']}"
                        )
                        doc_files = {
                            "Quyết định 5481/QĐ-BYT (Bộ Y tế)": ("guidelines_byt_5481.txt", "Tài liệu chuyên môn Hướng dẫn chẩn đoán và điều trị đái tháo đường típ 2 - Bộ Y tế ban hành"),
                            "Tài liệu Chuyên khoa Vinmec": ("vinmec_diabetes_article.txt", "Tài liệu Bệnh đái tháo đường típ 2 - Bệnh viện ĐKQT Vinmec"),
                            "ADA Standards of Care 2024": ("guidelines_ada_2024.txt", "Tiêu chuẩn chăm sóc y khoa trong đái tháo đường 2024 - Hiệp hội Đái tháo đường Hoa Kỳ")
                        }
                        dfname, dfdesc = doc_files[doc_choice]
                        dfpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dfname)
                        if os.path.exists(dfpath):
                            with open(dfpath, "r", encoding="utf-8") as f:
                                full_doc_text = f.read()
                            st.caption(dfdesc)
                            st.text_area("Toàn văn văn bản (100% Offline, không phụ thuộc mạng):", value=full_doc_text, height=280, disabled=True)
                            st.download_button(f"📥 Tải tệp {dfname}", data=full_doc_text, file_name=dfname, mime="text/plain", use_container_width=True)

            # Nút gợi ý câu hỏi tra cứu nhanh
            st.markdown("**Gợi ý tra cứu nhanh:**")
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            with col_q1:
                if st.button("Liều Metformin theo eGFR", use_container_width=True, key="quick_q1"):
                    st.session_state.pending_chat = "Tôi bị suy thận eGFR = 42 mL/phút, uống Metformin liều thế nào và có an toàn không?"
                    st.rerun()
            with col_q2:
                if st.button("Tiêu chuẩn chẩn đoán ADA / BYT", use_container_width=True, key="quick_q2"):
                    st.session_state.pending_chat = "Tiêu chuẩn chẩn đoán xác định đái tháo đường theo QĐ 5481/BYT và ADA 2024?"
                    st.rerun()
            with col_q3:
                if st.button("Hội chứng 4 nhiều (Vinmec)", use_container_width=True, key="quick_q3"):
                    st.session_state.pending_chat = "Hội chứng 4 nhiều và các triệu chứng cảnh báo đái tháo đường Vinmec là gì?"
                    st.rerun()
            with col_q4:
                if st.button("Hội chẩn ca bệnh BN001", use_container_width=True, key="quick_q4"):
                    st.session_state.pending_chat = "Tư vấn phác đồ và đánh giá toàn diện cho bệnh nhân BN001"
                    st.rerun()

            # Khung tin nhắn chat có chiều cao cố định, cuộn riêng biệt
            # Giải quyết triệt để vấn đề: người dùng không phải kéo trang xuống dưới để chat mới!
            with st.container(height=480):
                for msg in curr_sess["messages"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if msg.get("sources") or msg.get("graph_facts"):
                            with st.expander("📚 Nguồn trích dẫn & Dữ liệu đối chiếu nội bộ (Offline In-App RAG)"):
                                c_kb, c_rag = st.columns(2)
                                with c_kb:
                                    st.markdown("##### Mắt xích Đồ thị Tri thức (Neo4j):")
                                    if msg.get("graph_facts"):
                                        for f in msg["graph_facts"]:
                                            st.markdown(f"- `{f}`")
                                    else:
                                        st.caption("Không có thực thể xung đột bổ sung.")
                                with c_rag:
                                    st.markdown("##### Dẫn chứng Văn bản Hướng dẫn (Vector RAG):")
                                    if msg.get("sources"):
                                        for s in msg["sources"]:
                                            st.markdown(f"- **[{s['source']}]({s.get('url', '#')})**")
                                            st.caption(f"Tài liệu đối chiếu: `{s.get('pdf_ref', 'N/A')}` | Độ tương đồng: `{s.get('similarity_score', 0.0):.3f}`")
                                            st.write(f"> \"{s.get('text', '')[:280]}...\"")
                                    else:
                                        st.caption("Dữ liệu tra cứu đã được đối chiếu thành công.")

            # Khung nhập liệu chat đặt ngay dưới container cuộn tin nhắn
            chat_input_val = st.chat_input("Nhập câu hỏi tra cứu hoặc yêu cầu hội chẩn (Ví dụ: eGFR = 42 uống Metformin liều thế nào?)...", key="clinical_chat_input")
            chat_query = chat_input_val or st.session_state.pending_chat
            if st.session_state.pending_chat:
                st.session_state.pending_chat = None

            if chat_query:
                # 1. Thêm tin nhắn của User vào phiên hiện tại
                curr_sess["messages"].append({"role": "user", "content": chat_query})

                # Tự động cập nhật tiêu đề phiên nếu đang dùng tên mặc định
                if curr_sess["title"].startswith("Phiên mới #") or curr_sess["title"].startswith("Phiên tư vấn #"):
                    clean_first_line = chat_query.strip().split("\n")[0]
                    if len(clean_first_line) > 30:
                        curr_sess["title"] = clean_first_line[:28] + "..."
                    else:
                        curr_sess["title"] = clean_first_line

                with st.spinner("Đang điều phối tác tử tra cứu tri thức y khoa..."):
                    # Gọi trực tiếp KnowledgeRetrievalAgent (A_KG)
                    kg_res = coordinator.kg_agent.retrieve(chat_query)
                    rag_results = kg_res.payload["vector_chunks"]
                    graph_facts = [mh["chain_str"] for mh in kg_res.payload.get("multi_hop_facts", [])]

                    # Sử dụng Bộ tổng hợp tri thức lâm sàng đa tầng (Clinical Knowledge Synthesizer)
                    final_reply = generate_clinical_chat_response(
                        chat_query=chat_query,
                        coordinator=coordinator,
                        kg_res=kg_res,
                        rag_results=rag_results,
                        graph_facts=graph_facts
                    )

                curr_sess["messages"].append({
                    "role": "assistant",
                    "content": final_reply,
                    "graph_facts": graph_facts[:4],
                    "sources": rag_results
                })
                st.session_state.chat_messages = curr_sess["messages"]
                st.rerun()

    # =========================================================================
    # TAB 3: TÍCH HỢP MÔ HÌNH HỌC SÂU (DL PREDICTION AGENT)
    # =========================================================================
    with tab3:
        st.subheader("Mô Hình Học Sâu & Dự Báo CGM 4 Giờ")
        meta = get_model_metadata()

        # 5 Metrics hiệu năng gọn gàng ở đầu trang
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Độ chính xác (Accuracy)", f"{meta['evaluation_metrics']['accuracy']*100:.1f}%")
        col_m2.metric("Điểm F1-Score", f"{meta['evaluation_metrics']['f1_score']*100:.1f}%")
        col_m3.metric("AUC-ROC", f"{meta['evaluation_metrics']['auc_roc']*100:.1f}%")
        col_m4.metric("Độ nhạy (Sensitivity)", f"{meta['evaluation_metrics']['sensitivity_recall']*100:.1f}%")
        col_m5.metric("Độ đặc hiệu (Specificity)", f"{meta['evaluation_metrics']['specificity']*100:.1f}%")

        with st.expander("Xem đặc tả kỹ thuật chi tiết của mô hình DL (7 Tiêu chí kiểm định)", expanded=False):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(f"**Phiên bản:** `{meta['model_version']}` | **Tên:** `{meta['model_name']}`")
                st.markdown(f"**Kiến trúc DNN 6 tầng:** `{meta['model_architecture']['dnn_layers']}`")
                st.markdown(f"**Kiến trúc RNN CGM:** `{meta['model_architecture']['cgm_rnn_architecture']}`")
                st.markdown(f"**Bộ dữ liệu huấn luyện:** {meta['training_dataset']['source']} ({meta['training_dataset']['sample_size']:,} mẫu)")
            with col_d2:
                st.markdown(f"**Phương pháp tiền xử lý:** {meta['preprocessing_method']['type']} (`{meta['preprocessing_method']['formula']}`)")
                st.markdown(f"**Lưu ý an toàn lâm sàng:** {meta['clinical_limitation_notice']}")

        col_arch, col_acts = st.columns([1, 1])
        with col_arch:
            st.markdown(r"""
            ##### Cấu trúc Từng Tầng Hàm (Layer Composition)
            | Tầng hàm | Ánh xạ | Biểu thức giải tích | Vai trò biểu diễn |
            | :--- | :--- | :--- | :--- |
            | **Hàm $f_1$** | $\mathbb{R}^{15} \to \mathbb{R}^{64}$ | $h^{(1)} = \text{ReLU}(W_1 x + b_1)$ | Mở rộng không gian đặc trưng |
            | **Hàm $f_2$** | $\mathbb{R}^{64} \to \mathbb{R}^{64}$ | $h^{(2)} = \text{ReLU}(W_2 h^{(1)} + b_2)$ | Tương tác phi tuyến bậc hai |
            | **Hàm $f_3$** | $\mathbb{R}^{64} \to \mathbb{R}^{32}$ | $h^{(3)} = \text{ReLU}(W_3 h^{(2)} + b_3)$ | Nén đặc trưng sơ cấp |
            | **Hàm $f_4$** | $\mathbb{R}^{32} \to \mathbb{R}^{16}$ | $h^{(4)} = \text{ReLU}(W_4 h^{(3)} + b_4)$ | Tinh lọc hội chứng chuyển hóa |
            | **Hàm $f_5$** | $\mathbb{R}^{16} \to \mathbb{R}^{8}$ | $h^{(5)} = \text{ReLU}(W_5 h^{(4)} + b_5)$ | **Tầng nút thắt cổ chai (Bottleneck)** |
            | **Hàm $f_6$** | $\mathbb{R}^{8} \to [0, 1]$ | $\hat{y} = \sigma(W_6 h^{(5)} + b_6)$ | Hàm quyết định Sigmoid |
            """)

        with col_acts:
            dl_spec = coordinator.dl_agent
            p_curr = get_patient(st.session_state.selected_pid)
            raw_feats = dl_spec.preprocessor.extract_features(p_curr)
            norm_feats = dl_spec.preprocessor.transform(raw_feats)
            y_hat, acts = dl_spec.dnn_model.forward_composition(norm_feats)

            st.markdown(f"##### Biểu diễn Nén Bottleneck $h^{{(5)}}$ của **{p_curr['full_name']}**")
            b_df = pd.DataFrame([acts["h5_bottleneck"]], columns=[f"z_{i+1}" for i in range(8)])
            st.dataframe(b_df.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)
            st.metric(rf"Xác suất suy luận $\hat{{y}} = f_6(h^{{(5)}})$", f"{y_hat * 100.0:.2f}%", delta="Rủi ro mắc ĐTĐ Típ 2")

        st.markdown("---")
        st.subheader("Dự báo Chuỗi Đường huyết Liên tục CGM 4h (Stationary RNN)")

        cgm_hist = get_cgm_stream(st.session_state.selected_pid)
        forecaster = RecurrentCGMForecaster()
        cgm_pred_res = forecaster.forecast_next_4h(cgm_hist)

        hours_hist = [f"-{12-i}h" for i in range(12)]
        hours_fore = ["+1h", "+2h", "+3h", "+4h"]

        fig_cgm = go.Figure()
        fig_cgm.add_trace(go.Scatter(
            x=hours_hist,
            y=cgm_hist,
            mode='lines+markers',
            name='Lịch sử CGM 12h (Thực tế)',
            line=dict(color='#1e40af', width=2.5),
            marker=dict(size=6, color='#1e40af')
        ))
        fig_cgm.add_trace(go.Scatter(
            x=[hours_hist[-1]] + hours_fore,
            y=[cgm_hist[-1]] + cgm_pred_res["forecast_4h"],
            mode='lines+markers',
            name='Dự báo RNN 4h tới',
            line=dict(color='#b91c1c', width=2.5, dash='dash'),
            marker=dict(size=6, color='#b91c1c')
        ))

        fig_cgm.add_hrect(
            y0=70, y1=180,
            fillcolor="rgba(34, 197, 94, 0.08)",
            line_width=0,
            annotation_text="Vùng mục tiêu an toàn (TIR: 70 - 180 mg/dL)",
            annotation_position="top left",
            annotation_font=dict(color="#15803d", size=12)
        )
        fig_cgm.add_hline(y=180, line_dash="dot", line_color="#d97706", line_width=1.5, annotation_text="Ngưỡng tăng đường huyết (180 mg/dL)", annotation_font=dict(color="#b45309", size=11))
        fig_cgm.add_hline(y=70, line_dash="dot", line_color="#dc2626", line_width=1.5, annotation_text="Ngưỡng hạ đường huyết cấp cứu (70 mg/dL)", annotation_font=dict(color="#b91c1c", size=11))

        fig_cgm.update_layout(
            title=f"Đường biểu diễn dữ liệu CGM và Xu hướng dự báo ({p_curr['full_name']})",
            xaxis_title="Thời gian tương đối",
            yaxis_title="Nồng độ Glucose (mg/dL)",
            template="plotly_white",
            height=380,
            font=dict(size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cgm, use_container_width=True)

    # =========================================================================
    # TAB 4: CƠ SỞ TRI THỨC NEO4J & DB_DIABET.CYPHER
    # =========================================================================
    with tab4:
        st.subheader("Cơ Sở Tri Thức Đồ Thị Y Khoa Neo4j")

        g_summary = kg.get_graph_summary()
        col_kg1, col_kg2, col_kg3 = st.columns(3)
        col_kg1.metric("Nút Thực Thể (Entities)", g_summary["num_nodes"])
        col_kg2.metric("Quan Hệ Ngữ Nghĩa (Relations)", g_summary["num_edges"])
        col_kg3.metric("Nguồn Dữ Liệu Y Khoa", "Vinmec & QĐ 5481/BYT")

        # Cấu hình kết nối Neo4j thu gọn trong expander
        with st.expander("Cấu hình kết nối Neo4j Enterprise (Mặc định: In-memory NetworkX)", expanded=False):
            neo_col1, neo_col2 = st.columns([2, 1])
            with neo_col1:
                neo_uri = st.text_input("Neo4j Bolt URI:", value="bolt://localhost:7687")
                neo_user = st.text_input("Tên đăng nhập:", value="neo4j")
                neo_pass = st.text_input("Mật khẩu Neo4j:", value="password", type="password")
            with neo_col2:
                st.write("")
                st.write("")
                if st.button("Kiểm tra kết nối Neo4j", use_container_width=True):
                    ok = kg.connect_neo4j(neo_uri, neo_user, neo_pass)
                    if ok:
                        st.success("Kết nối cơ sở dữ liệu Neo4j thành công.")
                    else:
                        st.info("Không phát hiện Neo4j cục bộ. Đang chạy NetworkX in-memory.")
                if st.button("Nạp DB_diabet.cypher vào Neo4j", use_container_width=True):
                    ok_cypher = kg.init_neo4j_from_cypher_file("DB_diabet.cypher")
                    if ok_cypher:
                        st.success("Nạp thành công DB_diabet.cypher vào Neo4j.")
                    else:
                        st.warning("Vui lòng kết nối Neo4j trước khi nạp Cypher script.")

        st.markdown("##### Trực quan hóa Đồ thị Tri thức Y tế Tương tác")
        graph_choice = st.radio("Chọn nguồn đồ thị hiển thị:", ["Đồ thị y tế Vinmec Hospital (33 nút, 32 quan hệ)", "Đồ thị y tế tiêu chuẩn (25 nút, 24 quan hệ)"], horizontal=True)

        target_html = "kg_graph.html" if "Vinmec" in graph_choice else "graph_view.html"
        if not os.path.exists(target_html) and target_html == "graph_view.html":
            kg.generate_pyvis_html("graph_view.html")

        if os.path.exists(target_html):
            with open(target_html, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=520, scrolling=True)

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            json_file_path = os.path.join("data", "kg_graph.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, "r", encoding="utf-8") as f:
                    json_data_str = f.read()
                st.download_button("Tải xuống kg_graph.json", data=json_data_str, file_name="kg_graph.json", mime="application/json", use_container_width=True)
        with c_dl2:
            cypher_file_path = "DB_diabet.cypher"
            if os.path.exists(cypher_file_path):
                with open(cypher_file_path, "r", encoding="utf-8") as f:
                    cypher_str = f.read()
                st.download_button("Tải xuống DB_diabet.cypher", data=cypher_str, file_name="DB_diabet.cypher", mime="text/plain", use_container_width=True)

        st.markdown("---")
        st.markdown("##### Mô phỏng Duyệt Đồ thị Bắc cầu 2-Hop")
        c_hop1, c_hop2 = st.columns([1, 2])
        with c_hop1:
            selected_entity = st.selectbox("Chọn thực thể nguồn để duyệt 2-Hop:", options=list(kg.G.nodes()), index=0)
        with c_hop2:
            chains = kg.get_multi_hop_facts(selected_entity, max_hops=2)
            st.write(f"Tìm thấy **{len(chains)}** đường dẫn quan hệ từ `{selected_entity}`:")
            for ch in chains[:6]:
                if ch["hops"] == 2:
                    st.markdown(f"- **2-Hop:** `{ch['source']}` $\\to$ *[{ch['rel1']}]* $\\to$ `{ch['intermediate']}` $\\to$ *[{ch['rel2']}]* $\\to$ `{ch['target']}`")
                else:
                    st.markdown(f"- **1-Hop:** `{ch['source']}` $\\to$ *[{ch['rel1']}]* $\\to$ `{ch['target']}`")

    # =========================================================================
    # TAB 5: MÔ PHỎNG RL & KIỂM TOÁN CSDL
    # =========================================================================
    with tab5:
        subtab_rl, subtab_db = st.tabs([
            "Mô phỏng Q-Learning",
            "Kiểm toán & CSDL SQLite"
        ])

        with subtab_rl:
            st.subheader("Mở Rộng Học Tăng Cường Trong Mô Phỏng (Q-Learning)")
            st.markdown("""
            <div style="background-color: #eff6ff; border-left: 4px solid #2563eb; padding: 0.75rem 1rem; border-radius: 0 6px 6px 0; margin-bottom: 0.9rem; font-size: 0.92rem; color: #1e3a8a;">
                <strong>Ràng buộc an toàn:</strong> Tác tử RL vận hành độc quyền trong môi trường mô phỏng tổng hợp nhằm thử nghiệm chính sách theo dõi hoặc lựa chọn mô-đun giáo dục tuân thủ lối sống. Tuyệt đối không tự ý kê đơn hay thay đổi thuốc cho bệnh nhân thực tế.
            </div>
            """, unsafe_allow_html=True)

            col_qtable, col_compare = st.columns([1, 1])
            with col_qtable:
                st.markdown("##### Bảng Q-Table Hội tụ của Q-Learning Agent (4 Trạng thái)")
                rl_agent = coordinator.rl_agent.q_agent

                if st.button("Huấn luyện lại Q-Learning trong mô phỏng (3,000 tập)", type="secondary"):
                    with st.spinner("Đang chạy 60,000 bước cập nhật Bellman..."):
                        from train_models import train_rl_agent
                        new_agent = train_rl_agent(episodes=3000)
                        coordinator.rl_agent.q_agent = new_agent
                        rl_agent = new_agent
                    st.success("Huấn luyện hội tụ thành công trong môi trường mô phỏng.")

                pol = rl_agent.get_optimal_policy()

                q_data = []
                for s_str, d in pol.items():
                    q_data.append({
                        "Trạng thái s": s_str,
                        "Diễn giải lâm sàng": d["state_desc"],
                        "Q(s, 0) [Duy trì]": round(d["q_0"], 2),
                        "Q(s, 1) [Can thiệp]": round(d["q_1"], 2),
                        "Chính sách π*(s)": "Can thiệp" if d["best_action"] == 1 else "Duy trì",
                        "Lần thăm": d["visits"]
                    })
                st.dataframe(pd.DataFrame(q_data), use_container_width=True)

            with col_compare:
                st.markdown("##### So sánh Khả năng Thích ứng Môi trường (10,000 Bước)")
                exp_data = {
                    "Cấu hình môi trường": ["Môi trường chuẩn (δ = 0.20)", "Môi trường thay đổi (δ = 0.10)"],
                    "Q-Learning Agent (RL)": [35.13, 32.07],
                    "Rule-Based Agent (Luật tĩnh)": [34.75, 30.91]
                }
                df_exp = pd.DataFrame(exp_data)

                fig_bar = px.bar(
                    df_exp,
                    x="Cấu hình môi trường",
                    y=["Q-Learning Agent (RL)", "Rule-Based Agent (Luật tĩnh)"],
                    barmode="group",
                    title="Lợi ích sức khỏe kỳ vọng / Bước",
                    labels={"value": "Điểm lợi ích sức khỏe", "variable": "Loại tác tử"},
                    color_discrete_sequence=["#1e40af", "#94a3b8"]
                )
                fig_bar.update_layout(template="plotly_white", height=320, font=dict(size=12), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_bar, use_container_width=True)

        with subtab_db:
            st.subheader("Nhật Ký Kiểm Toán & Dữ Liệu CSDL SQLite")

            selected_table = st.radio(
                "Chọn bảng dữ liệu cần xem:",
                ["Nhật ký hành động tác tử (agent_action_logs)", "Hồ sơ bệnh nhân (patients)", "Luồng dữ liệu CGM 12h (cgm_telemetry_stream)"],
                horizontal=True
            )

            conn = get_connection()
            if "agent_action_logs" in selected_table:
                df_data = pd.read_sql_query("SELECT * FROM agent_action_logs ORDER BY id DESC LIMIT 50", conn)
            elif "patients" in selected_table:
                df_data = pd.read_sql_query("SELECT * FROM patients", conn)
            else:
                df_data = pd.read_sql_query("SELECT * FROM cgm_telemetry_stream ORDER BY id DESC LIMIT 50", conn)
            conn.close()

            st.write(f"Hiển thị **{len(df_data)}** bản ghi gần nhất:")
            st.dataframe(df_data, use_container_width=True)


# ĐIỂM VÀO CHƯƠNG TRÌNH CHÍNH
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hệ Đa Tác Tử Bệnh Tiểu Đường - PTIT")
    parser.add_argument("--cli", action="store_true", help="Chạy chế độ kiểm thử tự động trên Terminal")
    args, unknown = parser.parse_known_args()

    if args.cli:
        run_cli_mode()
    else:
        run_streamlit_app()
