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
import html
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
    print("\n[INFO] Khởi tạo CSDL SQLite, Đồ thị Tri thức Neo4j (DB_diabet.cypher) và Vector RAG...")
    init_database()
    kg = MedicalKnowledgeGraph()
    kg_summary = kg.get_graph_summary()
    if kg.neo4j_connected:
        print(
            f"[INFO] Kết nối Neo4j mặc định thành công tại {kg.neo4j_uri} "
            f"({kg_summary['num_nodes']} nút, {kg_summary['num_edges']} quan hệ)."
        )
    else:
        print(
            f"[WARN] Chưa kết nối được Neo4j ({kg.neo4j_uri}): {kg.neo4j_error}. "
            f"Đang dùng fallback NetworkX ({kg_summary['num_nodes']} nút, {kg_summary['num_edges']} quan hệ)."
        )
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
    print("Dữ liệu đầu vào lỗi: Tuổi=-8, BMI=5.0, Glucose=3500 mg/dL, HA=70/110 mmHg, eGFR=-10")
    bad_result = coordinator.execute_workflow(invalid_patient, "Kiểm thử kiểm tra lỗi")
    print(f"Trạng thái điều phối: {bad_result['status'].upper()}")
    print("Các lỗi xác thực được PatientDataAgent phát hiện:")
    for err in bad_result.get("errors", []):
        print(f"  [X] {err}")
    print("[KẾT LUẬN KIỂM THỬ] Coordinator Agent đã chấm dứt quy trình ngay tại Bước 5 để bảo đảm an toàn.")
    print("Để mở giao diện trực quan, chạy lệnh: streamlit run diabetes_multi_agent_app.py\n")


# =============================================================================
# CHẾ ĐỘ 2: GIAO DIỆN WEB (STREAMLIT - THAM KHẢO GEMINI WEB)
# =============================================================================
def generate_clinical_chat_response(chat_query: str, coordinator, kg_res, rag_results, graph_facts) -> str:
    """
    Tổng hợp câu trả lời lâm sàng trực tiếp, ngắn gọn và rõ ràng từ GraphRAG, Vector RAG và hệ đa tác tử.
    """
    lower_q = chat_query.lower()

    # 1. TƯ VẤN CA BỆNH / HỘI CHẨN
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

        return (
            f"### Tư vấn lâm sàng `{p_info['patient_id']}` — {p_info['full_name']}\n"
            f"Tuổi `{p_info['age']}` · BMI `{p_info['bmi']}` kg/m² · `eGFR {p_info['egfr']}` mL/phút/1.73m²\n\n"
            f"#### 1. Nguy cơ & dự báo đường huyết\n"
            f"- **Xác suất ĐTĐ Típ 2:** **{dl_out['risk_analysis']['risk_percentage']}%** ({dl_out['risk_analysis']['risk_level']})\n"
            f"- **Dự báo CGM 4h tới:** `{dl_out['cgm_forecast']['forecast_4h']}` mg/dL (Đỉnh **{dl_out['cgm_forecast']['peak_glucose']} mg/dL**, mục tiêu sau ăn `< 180 mg/dL`)\n\n"
            f"#### 2. Dược lý & an toàn thận\n"
            f"- **Chỉnh liều Metformin:** {rag_out['safety_check']['dose_adjustment']}\n"
            f"- **Thuốc phối hợp:** {', '.join(rag_out['safety_check']['recommended_drugs'])}\n"
            f"- **Bảo vệ cầu thận:** Ưu tiên nhóm `ACEi` hoặc `ARB` nếu kèm tăng huyết áp hoặc đạm niệu.\n\n"
            f"#### 3. Can thiệp lối sống\n"
            f"- **Chính sách khuyến nghị:** {rl_out['action_desc']}\n"
            f"{rl_out['detailed_plan']}\n\n"
            f"*Kiểm toán `{agent_res['request_id']}` · Groundedness `{agent_res['groundedness_score']}%` · Xử lý `{agent_res['latency_ms']} ms`*"
        )

    # 2. BIẾN CHỨNG ĐÁI THÁO ĐƯỜNG
    if any(k in lower_q for k in ["biến chứng", "hậu quả", "nguy hiểm", "mù", "loét", "cắt cụt", "đoạn chi", "dka", "hhs", "tai biến", "đột quỵ"]):
        return (
            "Biến chứng đái tháo đường típ 2 chia thành 2 nhóm chính:\n\n"
            "### Biến chứng cấp tính\n"
            "- **Hạ đường huyết (`< 70 mg/dL`):** Run tay, vã mồ hôi, có thể co giật hoặc hôn mê nếu dùng quá liều thuốc/insulin hoặc bỏ bữa.\n"
            "- **Nhiễm toan ceton (`DKA`):** Đường huyết thường `> 250 mg/dL`, hơi thở mùi aceton, toan chuyển hóa, rối loạn điện giải.\n"
            "- **Tăng áp lực thẩm thấu (`HHS`):** Glucose máu `> 600 mg/dL`, mất nước nặng, lơ mơ, nguy cơ tử vong cao ở người lớn tuổi.\n\n"
            "### Biến chứng mạn tính\n"
            "- **Thận:** Tiểu đạm vi thể tiến triển đến suy thận mạn.\n"
            "- **Mắt:** Tổn thương vi mạch võng mạc, phù hoàng điểm, nguy cơ giảm thị lực hoặc mù lòa.\n"
            "- **Thần kinh ngoại biên:** Tê bì, châm chích, mất cảm giác đau ở bàn chân.\n"
            "- **Tim mạch & đột quỵ:** Tăng nguy cơ xơ vữa mạch, nhồi máu cơ tim và tai biến mạch máu não.\n"
            "- **Bàn chân đái tháo đường:** Loét và nhiễm trùng khó lành, có thể dẫn tới đoạn chi.\n\n"
            "### Tầm soát định kỳ\n"
            "- Duy trì `HbA1c < 7.0%` và huyết áp `< 130/80 mmHg`.\n"
            "- Kiểm tra `eGFR`, `uACR` mỗi 6–12 tháng; soi đáy mắt hàng năm và kiểm tra bàn chân mỗi ngày."
        )

    # 3. NGUYÊN NHÂN & CƠ CHẾ BỆNH SINH
    if any(k in lower_q for k in ["nguyên nhân", "bệnh sinh", "cơ chế", "tại sao bị", "kháng insulin", "beta tụy", "yếu tố nguy cơ", "yếu tố di truyền"]):
        return (
            "Đái tháo đường típ 2 hình thành từ hai cơ chế cốt lõi:\n\n"
            "1. **Kháng insulin:** Tế bào gan, cơ xương và mô mỡ giảm đáp ứng với insulin; gan tiếp tục phóng thích glucose trong khi mô cơ giảm thu nhận đường.\n"
            "2. **Suy giảm tế bào beta đảo tụy:** Tăng đường huyết và acid béo tự do kéo dài gây ngộ độc đường/mỡ, làm tế bào beta suy giảm dần khả năng tiết insulin.\n\n"
            "### Yếu tố nguy cơ thường gặp\n"
            "- **Thừa cân, béo phì:** `BMI >= 23 kg/m²` (người châu Á) hoặc vòng eo lớn (`>= 90 cm` ở nam, `>= 80 cm` ở nữ).\n"
            "- **Ít vận động:** Giảm độ nhạy insulin của cơ xương.\n"
            "- **Tiền sử gia đình:** Bố, mẹ hoặc anh chị em ruột mắc đái tháo đường.\n"
            "- **Bệnh đồng mắc:** Tăng huyết áp (`>= 140/90 mmHg`), rối loạn lipid máu, tiền sử đái tháo đường thai kỳ.\n"
            "- **Dấu gai đen:** Vùng da sẫm dày ở nếp gấp cổ, nách gợi ý kháng insulin."
        )

    # 4. TRIỆU CHỨNG LÂM SÀNG (HỘI CHỨNG 4 NHIỀU)
    if any(k in lower_q for k in ["4 nhiều", "triệu chứng", "dấu hiệu", "biểu hiện", "tiểu nhiều", "khát nước", "sụt cân", "mau đói", "cảnh báo"]):
        return (
            "**Hội chứng 4 nhiều** là nhóm triệu chứng kinh điển của tăng đường huyết:\n\n"
            "1. **Tiểu nhiều:** Khi glucose máu vượt ngưỡng thận (`~180 mg/dL`), đường dư thải qua nước tiểu kéo theo nước theo cơ chế lợi niệu thẩm thấu, gây tiểu đêm và lượng nước tiểu tăng (`3–5 lít/ngày`).\n"
            "2. **Khát nhiều:** Mất nước làm tăng áp lực thẩm thấu huyết tương, kích thích trung tâm khát liên tục.\n"
            "3. **Ăn nhiều:** Tế bào thiếu insulin không hấp thu được glucose để tạo năng lượng nên phát tín hiệu đói kéo dài.\n"
            "4. **Sụt cân nhanh:** Cơ thể phải dị hóa mô mỡ và cơ bắp để bù năng lượng, gây giảm `3–8 kg` trong thời gian ngắn.\n\n"
            "### Dấu hiệu đi kèm\n"
            "- Mắt nhìn mờ tạm thời do thay đổi áp lực thẩm thấu thủy tinh thể.\n"
            "- Mệt mỏi kéo dài, vết thương lâu lành, tê bì đầu ngón tay chân."
        )

    # 5. TIÊU CHUẨN CHẨN ĐOÁN & CHỈ SỐ XÉT NGHIỆM
    if any(k in lower_q for k in ["chẩn đoán", "tiêu chuẩn", "xét nghiệm", "chỉ số", "hba1c", "fpg", "ogtt", "đường huyết đói", "nghiệm pháp"]):
        return (
            "Theo **Quyết định 5481/QĐ-BYT** và **ADA 2024**, chẩn đoán đái tháo đường khi thỏa mãn ít nhất **1 trong 4 tiêu chí** trên mẫu máu tĩnh mạch:\n\n"
            "| Xét nghiệm | Đái tháo đường | Tiền đái tháo đường | Bình thường | Điều kiện |\n"
            "| :--- | :--- | :--- | :--- | :--- |\n"
            "| **Glucose đói (`FPG`)** | $\\ge 126$ mg/dL (`7.0 mmol/L`) | `100 – 125` mg/dL | `< 100` mg/dL | Nhịn ăn ít nhất 8 giờ |\n"
            "| **`HbA1c`** | $\\ge 6.5\\%$ | `5.7 – 6.4%` | `< 5.7%` | Chuẩn hóa NGSP/DCCT |\n"
            "| **`OGTT` sau 2 giờ** | $\\ge 200$ mg/dL (`11.1 mmol/L`) | `140 – 199` mg/dL | `< 140` mg/dL | Uống 75g glucose |\n"
            "| **Glucose ngẫu nhiên** | $\\ge 200$ mg/dL | — | — | Kèm triệu chứng 4 nhiều |\n\n"
            "### Lưu ý thực hành\n"
            "- Nếu không có triệu chứng tăng đường huyết rõ, cần lặp lại xét nghiệm lần hai vào ngày khác để xác định.\n"
            "- Không dùng máy đo đường huyết mao mạch cầm tay để chẩn đoán xác định.\n"
            "- Mục tiêu kiểm soát phổ biến: `HbA1c < 7.0%`, `TIR (70–180 mg/dL) > 70%`."
        )

    # 6. DƯỢC LÝ, METFORMIN & AN TOÀN THẬN (eGFR)
    if any(k in lower_q for k in ["metformin", "suy thận", "egfr", "thuốc", "dược lý", "uống thuốc", "chỉnh liều", "insulin", "sglt2", "empagliflozin", "dapagliflozin"]):
        return (
            "`Metformin` là thuốc lựa chọn đầu tay trong điều trị đái tháo đường típ 2, nhưng liều dùng bắt buộc phải điều chỉnh theo độ lọc cầu thận `eGFR`:\n\n"
            "1. **`eGFR >= 45 mL/phút/1.73m²`:** Dùng liều tiêu chuẩn (tối đa `2000 mg/ngày`), theo dõi `eGFR` mỗi năm.\n"
            "2. **`eGFR 30 – 44 mL/phút/1.73m²` (Suy thận độ 3b):** Giảm **50% liều tối đa** (không quá `1000 mg/ngày`), không khởi đầu mới ở ngưỡng này và kiểm tra `eGFR` mỗi `3–6` tháng.\n"
            "3. **`eGFR < 30 mL/phút/1.73m²` (Suy thận độ 4–5):** **Chống chỉ định tuyệt đối** do nguy cơ nhiễm toan acid lactic; ngừng `Metformin` và chuyển sang `Insulin` hoặc thuốc thải trừ ngoài thận.\n\n"
            "### Thuốc phối hợp bảo vệ tim – thận\n"
            "- **Nhóm `SGLT2i` (`Empagliflozin`, `Dapagliflozin`):** Làm chậm suy giảm `eGFR` và giảm biến cố suy tim.\n"
            "- **Nhóm `ACEi` / `ARB`:** Ưu tiên khi kèm tăng huyết áp hoặc albumin niệu."
        )

    # 7. DINH DƯỠNG, TẬP LUYỆN & LỐI SỐNG
    if any(k in lower_q for k in ["dinh dưỡng", "ăn uống", "vận động", "tập luyện", "lối sống", "tinh bột", "thực đơn", "kiêng", "thể dục", "chế độ ăn"]):
        rl_res = coordinator.rl_agent.recommend_simulation_policy(peak_cgm=195.0, previous_compliance=0)
        return (
            "### Dinh dưỡng y khoa\n"
            "- Giảm `30–40%` tinh bột hấp thu nhanh (cơm trắng, bún, bánh mì trắng, đồ uống ngọt) để hạn chế đỉnh đường huyết sau ăn.\n"
            "- Tăng chất xơ hòa tan và tinh bột phức hợp (gạo lứt, yến mạch, các loại đậu, rau xanh đậm).\n"
            "- Bổ sung đạm nạc (cá, ức gà, trứng, đậu hũ) và chất béo không bão hòa (dầu ô liu, bơ, hạt).\n\n"
            "### Vận động thể lực\n"
            "- Duy trì tối thiểu `150 phút/tuần` vận động nhịp điệu cường độ trung bình, không nghỉ quá 2 ngày liên tiếp.\n"
            "- **Đi bộ sau ăn 15–30 phút:** Kéo dài `20–30 phút` giúp kích hoạt kênh `GLUT4` đưa glucose vào cơ độc lập với insulin.\n\n"
            "### Gợi ý từ mô hình mô phỏng RL\n"
            f"- **{rl_res.payload['action_description']}**\n"
            f"{rl_res.payload['detailed_plan']}"
        )

    # 8. HẠ ĐƯỜNG HUYẾT & XỬ TRÍ CẤP CỨU
    if any(k in lower_q for k in ["hạ đường huyết", "tụt đường huyết", "cấp cứu", "run tay", "vã mồ hôi", "xỉu", "choáng", "đường huyết thấp"]):
        return (
            "Hạ đường huyết xảy ra khi glucose máu `< 70 mg/dL` (`< 3.9 mmol/L`), thường gặp ở người dùng `Insulin` hoặc `Sulfonylurea`.\n\n"
            "### Dấu hiệu nhận biết\n"
            "- **Sớm:** Run tay, vã mồ hôi lạnh, tim đập nhanh, đói cồn cào, lo âu.\n"
            "- **Muộn:** Chóng mặt, nhìn mờ, lú lẫn, nói khó, co giật hoặc hôn mê.\n\n"
            "### Quy tắc xử trí `15 - 15`\n"
            "1. **Nạp ngay `15g` đường hấp thu nhanh:** `1/2` ly nước ép/nước ngọt có đường (`100–150 mL`), hoặc `3` thìa cà phê đường pha nước ấm, hoặc `3–4` viên kẹo ngọt.\n"
            "2. **Nghỉ ngơi `15 phút`**, tránh vận động.\n"
            "3. **Đo lại đường huyết:** Nếu vẫn `< 70 mg/dL`, lặp lại bước 1; khi đạt `>= 70 mg/dL`, ăn thêm bữa phụ nhẹ.\n\n"
            "*Nếu người bệnh hôn mê hoặc không nuốt được, không đổ nước vào miệng mà cần cấp cứu y tế hoặc tiêm `Glucagon` ngay.*"
        )

    # 9. TRUY XUẤT CHUNG (HYBRID GRAPHRAG)
    parts = []
    if graph_facts:
        parts.append("**Liên kết tri thức liên quan:** " + " · ".join(f"`{f}`" for f in graph_facts[:3]))

    if rag_results:
        for chunk in rag_results[:2]:
            src = chunk.get("source", "Tài liệu y khoa")
            txt = chunk.get("text", "").strip()
            parts.append(f"**{src}**\n> {txt}")
    else:
        parts.append("Không tìm thấy trích đoạn khớp trực tiếp trong kho dữ liệu; bạn có thể hỏi cụ thể về chẩn đoán, `Metformin`, biến chứng hoặc mã bệnh nhân (`BN001`–`BN003`).")

    return "\n\n".join(parts)


def _render_gemini_sidebar_logo() -> str:
    return """
    <div class="gemini-brand">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <path d="M12 2C12.3 7.1 16.9 11.7 22 12C16.9 12.3 12.3 16.9 12 22C11.7 16.9 7.1 12.3 2 12C7.1 11.7 11.7 7.1 12 2Z" fill="url(#gemGrad)"/>
            <defs>
                <linearGradient id="gemGrad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#4285F4"/>
                    <stop offset="55%" stop-color="#9B72CB"/>
                    <stop offset="100%" stop-color="#D96570"/>
                </linearGradient>
            </defs>
        </svg>
        <span class="gemini-brand-text">Trợ lý Tiểu đường</span>
    </div>
    """


def _inject_gemini_styles(is_chat_view: bool):
    import streamlit as st

    max_w = "800px" if is_chat_view else "1120px"
    bottom_pad = "6.5rem" if is_chat_view else "3rem"

    st.markdown(f"""
    <style>
        /* Typography & base light surface */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background-color: #ffffff;
            color: #1f1f1f;
            font-family: "Google Sans", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}

        /* Hide noisy Streamlit top bar chrome while keeping sidebar toggle */
        [data-testid="stDecoration"],
        [data-testid="stAppDeployButton"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        footer {{
            display: none !important;
        }}
        [data-testid="stHeader"] {{
            background: transparent !important;
            backdrop-filter: none !important;
            border-bottom: none !important;
            height: 3rem !important;
            min-height: 3rem !important;
            pointer-events: none;
        }}
        [data-testid="stExpandSidebarButton"] {{
            pointer-events: auto;
            background: rgba(255, 255, 255, 0.92) !important;
            border-radius: 999px !important;
        }}

        /* Main container sizing */
        .main .block-container,
        [data-testid="stMainBlockContainer"] {{
            max-width: {max_w};
            padding-top: 4rem !important;
            padding-bottom: {bottom_pad} !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            margin: 0 auto;
        }}

        /* Gemini-style Left Sidebar */
        [data-testid="stSidebar"] {{
            background-color: #f0f4f9 !important;
            border-right: none !important;
        }}
        [data-testid="stSidebarHeader"] {{
            height: 3rem !important;
            min-height: 3rem !important;
            margin-bottom: 0.25rem !important;
            padding: 0.5rem 0.65rem 0 0.65rem !important;
        }}
        [data-testid="stSidebarContent"] {{
            padding-top: 0.25rem !important;
            padding-left: 0.6rem;
            padding-right: 0.6rem;
        }}
        [data-testid="stSidebarUserContent"] {{
            padding-top: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }}
        .gemini-brand {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.35rem 0.65rem 0.9rem 0.65rem;
        }}
        .gemini-brand-text {{
            font-size: 1.15rem;
            font-weight: 500;
            color: #1f1f1f;
            letter-spacing: -0.01em;
        }}
        .sidebar-section-label {{
            font-size: 0.78rem;
            font-weight: 500;
            color: #444746;
            padding: 0.65rem 0.85rem 0.25rem 0.85rem;
            margin-top: 0.25rem;
        }}
        .sidebar-status-row {{
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.8rem;
            color: #444746;
            padding: 0.35rem 0.4rem;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }}

        /* Sidebar flat pill buttons */
        [data-testid="stSidebar"] button[kind="tertiary"],
        [data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"],
        [data-testid="stSidebar"] button[kind="secondary"],
        [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {{
            width: 100%;
            justify-content: flex-start !important;
            text-align: left !important;
            border-radius: 999px !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0.42rem 0.85rem !important;
            min-height: 36px !important;
            font-size: 0.88rem !important;
            line-height: 1.3 !important;
            transition: background-color 0.15s ease;
        }}
        [data-testid="stSidebar"] button[kind="tertiary"],
        [data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] {{
            background-color: transparent !important;
            color: #1f1f1f !important;
            font-weight: 400 !important;
        }}
        [data-testid="stSidebar"] button[kind="tertiary"]:hover,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"]:hover {{
            background-color: #e2e7ee !important;
        }}
        [data-testid="stSidebar"] button[kind="secondary"],
        [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {{
            background-color: #d3e3fd !important;
            color: #041e49 !important;
            font-weight: 500 !important;
        }}
        [data-testid="stSidebar"] button[kind="secondary"]:hover,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {{
            background-color: #c2d7fa !important;
        }}
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child button,
        [data-testid="stSidebar"] [data-testid="stPopover"] button {{
            justify-content: center !important;
            padding: 0.25rem !important;
            width: 32px !important;
            min-width: 32px !important;
            height: 32px !important;
            min-height: 32px !important;
            border-radius: 999px !important;
            border: none !important;
            background-color: transparent !important;
            color: #444746 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child button:hover,
        [data-testid="stSidebar"] [data-testid="stPopover"] button:hover {{
            background-color: #e2e7ee !important;
            color: #1f1f1f !important;
        }}
        [data-testid="stSidebar"] button p {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        [data-testid="stSidebar"] hr {{
            margin: 0.55rem 0.5rem;
            border-color: #dde3ea;
        }}

        /* Main area buttons & subtle popovers */
        [data-testid="stMain"] button[kind="primary"],
        [data-testid="stMain"] button[data-testid="stBaseButton-primary"] {{
            background-color: #1a73e8 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 500 !important;
            padding: 0.45rem 1.25rem !important;
        }}
        [data-testid="stMain"] button[kind="primary"]:hover,
        [data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover {{
            background-color: #1557b0 !important;
        }}
        [data-testid="stMain"] button[kind="secondary"],
        [data-testid="stMain"] button[data-testid="stBaseButton-secondary"] {{
            background-color: #ffffff !important;
            color: #1f1f1f !important;
            border: 1px solid #dfe3e8 !important;
            border-radius: 999px !important;
            font-weight: 500 !important;
        }}
        [data-testid="stMain"] button[kind="secondary"]:hover,
        [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover {{
            background-color: #f0f4f9 !important;
            border-color: #c4cdd5 !important;
        }}
        [data-testid="stMain"] [data-testid="stPopover"] button {{
            border: none !important;
            border-radius: 999px !important;
            width: 36px !important;
            height: 36px !important;
            min-height: 36px !important;
            padding: 0.25rem !important;
            background-color: transparent !important;
        }}
        [data-testid="stMain"] [data-testid="stPopover"] button:hover {{
            background-color: #f0f4f9 !important;
        }}

        /* Gemini Chat Canvas */
        .gemini-greeting {{
            text-align: center;
            margin: 4.5rem auto 1.75rem auto;
            max-width: 600px;
        }}
        .gemini-greeting h1 {{
            font-size: 2.25rem;
            font-weight: 500;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #1a73e8 0%, #6b73e8 45%, #d96570 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 0.5rem 0;
        }}
        .gemini-greeting p {{
            font-size: 1rem;
            color: #5f6368;
            margin: 0;
        }}

        .gemini-user-row {{
            display: flex;
            justify-content: flex-end;
            margin: 1.5rem 0 1rem 0;
        }}
        .gemini-user-bubble {{
            background-color: #f0f4f9;
            color: #1f1f1f;
            border-radius: 24px 24px 6px 24px;
            padding: 0.78rem 1.25rem;
            max-width: 76%;
            font-size: 0.98rem;
            line-height: 1.55;
            word-wrap: break-word;
        }}

        /* Assistant markdown styling matching Gemini Web */
        [data-testid="stMain"] .stMarkdown {{
            color: #1f1f1f;
            font-size: 0.98rem;
            line-height: 1.65;
        }}
        [data-testid="stMain"] .stMarkdown code {{
            background-color: #f0f4f9;
            color: #1f1f1f;
            padding: 0.12rem 0.42rem;
            border-radius: 6px;
            font-size: 0.88em;
        }}
        [data-testid="stMain"] .stMarkdown pre code {{
            background-color: transparent;
            padding: 0;
        }}
        [data-testid="stMain"] .stMarkdown h3 {{
            font-size: 1.18rem;
            font-weight: 500;
            color: #1f1f1f;
            margin-top: 1.15rem;
            margin-bottom: 0.5rem;
        }}
        [data-testid="stMain"] .stMarkdown h4 {{
            font-size: 1.02rem;
            font-weight: 600;
            color: #1f1f1f;
            margin-top: 0.9rem;
            margin-bottom: 0.35rem;
        }}

        /* Clean minimal expanders */
        [data-testid="stExpander"] {{
            border: 1px solid #e7ebf0 !important;
            border-radius: 12px !important;
            background-color: #fafbfd !important;
            margin-top: 0.4rem;
            margin-bottom: 1rem;
        }}
        [data-testid="stExpander"] summary {{
            font-size: 0.86rem !important;
            color: #444746 !important;
            font-weight: 500 !important;
        }}

        /* Floating Pill Chat Input at Bottom */
        [data-testid="stBottom"] {{
            background: linear-gradient(to top, #ffffff 82%, rgba(255, 255, 255, 0)) !important;
            padding-top: 0.6rem !important;
        }}
        [data-testid="stBottomBlockContainer"] {{
            max-width: 800px;
            margin: 0 auto;
            padding-top: 0.35rem;
            padding-bottom: 0.45rem;
        }}
        [data-testid="stBottomBlockContainer"]::after {{
            content: "Thông tin chỉ mang tính hỗ trợ quyết định y khoa, không thay thế chẩn đoán của bác sĩ.";
            display: block;
            text-align: center;
            font-size: 0.75rem;
            color: #5f6368;
            margin-top: 0.45rem;
        }}
        [data-testid="stChatInput"] {{
            background-color: #ffffff !important;
            border: 1px solid #dfe3e8 !important;
            border-radius: 28px !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06) !important;
            padding: 0.25rem 0.55rem !important;
        }}
        [data-testid="stChatInput"]:focus-within {{
            border-color: #a8c7fa !important;
            box-shadow: 0 3px 16px rgba(26, 115, 232, 0.12) !important;
        }}

        /* Metric cards for clinical/analytics pages */
        div[data-testid="stMetric"] {{
            background-color: #f8fafd;
            border: 1px solid #e7ebf0;
            border-radius: 14px;
            padding: 0.8rem 1rem;
        }}
        div[data-testid="stMetricLabel"] p {{
            font-size: 0.82rem !important;
            color: #5f6368 !important;
            font-weight: 500;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.45rem !important;
            color: #1f1f1f !important;
            font-weight: 500;
        }}

        /* Page titles for non-chat views */
        .view-title {{
            font-size: 1.45rem;
            font-weight: 500;
            color: #1f1f1f;
            letter-spacing: -0.01em;
            margin: 0 0 0.85rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)


def run_streamlit_app():
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px

    st.set_page_config(
        page_title="Trợ lý Tiểu đường",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    @st.cache_resource
    def load_system_core():
        init_database()
        kg = MedicalKnowledgeGraph()
        vstore = ClinicalVectorStore()
        vstore.build_index_from_directory()
        coord = CoordinatorAgent(kg, vstore)
        return kg, vstore, coord

    kg, vstore, coordinator = load_system_core()

    # Session state initialization
    if "active_view" not in st.session_state:
        st.session_state.active_view = "chat"
    if "selected_pid" not in st.session_state:
        st.session_state.selected_pid = "BN001"
    if "pending_chat" not in st.session_state:
        st.session_state.pending_chat = None

    if "chat_sessions" not in st.session_state:
        legacy_msgs = st.session_state.get("chat_messages")
        first_sess_id = f"sess_{int(time.time() * 1000)}"
        st.session_state.chat_sessions = [
            {
                "id": first_sess_id,
                "title": "Cuộc trò chuyện mới",
                "created_at": time.strftime("%H:%M - %d/%m"),
                "messages": legacy_msgs if legacy_msgs else []
            }
        ]
        st.session_state.current_session_id = first_sess_id

    # Normalize any legacy broken URLs inside stored chat history
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

    _inject_gemini_styles(is_chat_view=(st.session_state.active_view == "chat"))

    # =========================================================================
    # SIDEBAR (GEMINI WEB STYLE)
    # =========================================================================
    with st.sidebar:
        st.markdown(_render_gemini_sidebar_logo(), unsafe_allow_html=True)

        if st.button("Cuộc trò chuyện mới", icon=":material/edit_square:", type="tertiary", width="stretch", key="nav_new_chat"):
            new_id = f"sess_{int(time.time() * 1000)}"
            st.session_state.chat_sessions.insert(0, {
                "id": new_id,
                "title": "Cuộc trò chuyện mới",
                "created_at": time.strftime("%H:%M - %d/%m"),
                "messages": []
            })
            st.session_state.current_session_id = new_id
            st.session_state.active_view = "chat"
            st.session_state.pending_chat = None
            st.rerun()

        nav_items = [
            ("consult", "Hội chẩn lâm sàng", ":material/stethoscope:"),
            ("dl", "Học sâu & CGM", ":material/monitoring:"),
            ("kg", "Đồ thị tri thức", ":material/hub:"),
            ("rl", "Mô phỏng RL & CSDL", ":material/database:"),
            ("docs", "Thư viện tài liệu", ":material/menu_book:"),
        ]
        for view_key, view_label, view_icon in nav_items:
            is_cur_view = (st.session_state.active_view == view_key)
            if st.button(
                view_label,
                icon=view_icon,
                type="secondary" if is_cur_view else "tertiary",
                width="stretch",
                key=f"nav_{view_key}"
            ):
                st.session_state.active_view = view_key
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">Gần đây</div>', unsafe_allow_html=True)

        for sess in st.session_state.chat_sessions:
            is_active_chat = (st.session_state.active_view == "chat" and sess["id"] == st.session_state.current_session_id)
            disp_title = sess["title"]
            if len(disp_title) > 26:
                disp_title = disp_title[:24] + "..."

            if len(st.session_state.chat_sessions) > 1:
                c_sbtn, c_sdel = st.columns([5.2, 1], gap="small")
                with c_sbtn:
                    if st.button(
                        disp_title,
                        key=f"btn_sess_{sess['id']}",
                        type="secondary" if is_active_chat else "tertiary",
                        width="stretch",
                        help=sess["title"]
                    ):
                        st.session_state.current_session_id = sess["id"]
                        st.session_state.active_view = "chat"
                        st.rerun()
                with c_sdel:
                    if st.button("", icon=":material/close:", key=f"del_sess_{sess['id']}", type="tertiary", help="Xóa cuộc trò chuyện"):
                        st.session_state.chat_sessions = [s for s in st.session_state.chat_sessions if s["id"] != sess["id"]]
                        if st.session_state.current_session_id == sess["id"]:
                            st.session_state.current_session_id = st.session_state.chat_sessions[0]["id"]
                        st.rerun()
            else:
                if st.button(
                    disp_title,
                    key=f"btn_sess_{sess['id']}",
                    type="secondary" if is_active_chat else "tertiary",
                    width="stretch",
                    help=sess["title"]
                ):
                    st.session_state.current_session_id = sess["id"]
                    st.session_state.active_view = "chat"
                    st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        dot_color = "#1e8e3e" if kg.neo4j_connected else "#f29900"
        dot_label = "Neo4j đã kết nối" if kg.neo4j_connected else "NetworkX (dự phòng)"
        c_stat, c_set = st.columns([4.5, 1])
        with c_stat:
            st.markdown(
                f'<div class="sidebar-status-row"><span class="status-dot" style="background:{dot_color};"></span><span>{dot_label}</span></div>',
                unsafe_allow_html=True
            )
        with c_set:
            with st.popover("", icon=":material/settings:", type="tertiary", help="Tùy chọn"):
                if st.button("Xóa toàn bộ cuộc trò chuyện", width="stretch", key="reset_all_chats"):
                    fresh_id = f"sess_{int(time.time() * 1000)}"
                    st.session_state.chat_sessions = [
                        {
                            "id": fresh_id,
                            "title": "Cuộc trò chuyện mới",
                            "created_at": time.strftime("%H:%M - %d/%m"),
                            "messages": []
                        }
                    ]
                    st.session_state.current_session_id = fresh_id
                    st.session_state.active_view = "chat"
                    st.rerun()

    # =========================================================================
    # VIEW 1: TRỢ LÝ HỘI THOẠI (GEMINI WEB CHAT)
    # =========================================================================
    if st.session_state.active_view == "chat":
        curr_sess = next(
            (s for s in st.session_state.chat_sessions if s["id"] == st.session_state.current_session_id),
            None
        )
        if not curr_sess:
            curr_sess = st.session_state.chat_sessions[0]
            st.session_state.current_session_id = curr_sess["id"]

        # Minimal top action bar (chỉ giữ tiêu đề nhẹ khi đã có chat + menu ⋮)
        c_top_l, c_top_r = st.columns([9, 1])
        with c_top_l:
            if curr_sess["messages"]:
                st.caption(curr_sess["title"])
        with c_top_r:
            with st.popover("", icon=":material/more_vert:", type="tertiary", help="Tùy chọn cuộc trò chuyện"):
                new_title_val = st.text_input("Đổi tên", value=curr_sess["title"], key=f"rename_input_{curr_sess['id']}")
                if st.button("Lưu tên", width="stretch", key=f"save_title_{curr_sess['id']}"):
                    if new_title_val.strip():
                        curr_sess["title"] = new_title_val.strip()
                        st.rerun()
                if st.button("Làm mới cuộc trò chuyện", width="stretch", key=f"clear_msgs_{curr_sess['id']}"):
                    curr_sess["messages"] = []
                    curr_sess["title"] = "Cuộc trò chuyện mới"
                    st.rerun()
                export_json = json.dumps({
                    "session_id": curr_sess["id"],
                    "title": curr_sess["title"],
                    "created_at": curr_sess.get("created_at"),
                    "messages": curr_sess["messages"]
                }, ensure_ascii=False, indent=2)
                st.download_button(
                    "Xuất JSON",
                    data=export_json,
                    file_name=f"{curr_sess['id']}.json",
                    mime="application/json",
                    width="stretch"
                )

        # Empty state vs conversation history
        if not curr_sess["messages"]:
            st.markdown("""
            <div class="gemini-greeting">
                <h1>Xin chào</h1>
                <p>Hỏi về tiêu chuẩn chẩn đoán, chỉnh liều Metformin theo eGFR, biến chứng hoặc hội chẩn ca bệnh.</p>
            </div>
            """, unsafe_allow_html=True)

            q_prompts = [
                ("Liều Metformin theo eGFR", "Tôi bị suy thận eGFR = 42 mL/phút, uống Metformin liều thế nào và có an toàn không?"),
                ("Tiêu chuẩn chẩn đoán", "Tiêu chuẩn chẩn đoán xác định đái tháo đường theo QĐ 5481/BYT và ADA 2024?"),
                ("Hội chứng 4 nhiều", "Hội chứng 4 nhiều và các triệu chứng cảnh báo đái tháo đường là gì?"),
                ("Hội chẩn ca bệnh BN001", "Tư vấn phác đồ và đánh giá toàn diện cho bệnh nhân BN001"),
            ]
            cols_q = st.columns(4, gap="small")
            for idx, (label, prompt_text) in enumerate(q_prompts):
                with cols_q[idx]:
                    if st.button(label, width="stretch", key=f"chip_{idx}"):
                        st.session_state.pending_chat = prompt_text
                        st.rerun()
        else:
            for msg in curr_sess["messages"]:
                if msg["role"] == "user":
                    safe_text = html.escape(msg["content"]).replace("\n", "<br>")
                    st.markdown(
                        f'<div class="gemini-user-row"><div class="gemini-user-bubble">{safe_text}</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(msg["content"])
                    if msg.get("sources") or msg.get("graph_facts"):
                        with st.expander("Nguồn tham khảo", expanded=False):
                            c_kb, c_rag = st.columns(2)
                            with c_kb:
                                st.markdown("**Đồ thị tri thức**")
                                if msg.get("graph_facts"):
                                    for f in msg["graph_facts"]:
                                        st.markdown(f"- `{f}`")
                                else:
                                    st.caption("Không có liên kết bổ sung.")
                            with c_rag:
                                st.markdown("**Tài liệu hướng dẫn**")
                                if msg.get("sources"):
                                    for s in msg["sources"]:
                                        st.markdown(f"- **[{s['source']}]({s.get('url', '#')})** (`{s.get('similarity_score', 0.0):.3f}`)")
                                        st.caption(f"{s.get('text', '')[:220]}...")
                                else:
                                    st.caption("Không có trích đoạn phù hợp.")

        chat_input_val = st.chat_input("Hỏi về bệnh tiểu đường hoặc nhập mã bệnh nhân...", key="clinical_chat_input")
        chat_query = chat_input_val or st.session_state.pending_chat
        if st.session_state.pending_chat:
            st.session_state.pending_chat = None

        if chat_query:
            curr_sess["messages"].append({"role": "user", "content": chat_query})

            if curr_sess["title"] in ("Cuộc trò chuyện mới", "Phiên tư vấn #1") or curr_sess["title"].startswith("Phiên mới #"):
                clean_first = chat_query.strip().split("\n")[0]
                curr_sess["title"] = (clean_first[:28] + "...") if len(clean_first) > 30 else clean_first

            with st.spinner("Đang tra cứu..."):
                kg_res = coordinator.kg_agent.retrieve(chat_query)
                rag_results = kg_res.payload["vector_chunks"]
                graph_facts = [mh["chain_str"] for mh in kg_res.payload.get("multi_hop_facts", [])]

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
    # VIEW 2: HỘI CHẨN LÂM SÀNG
    # =========================================================================
    elif st.session_state.active_view == "consult":
        st.markdown('<div class="view-title">Hội chẩn lâm sàng</div>', unsafe_allow_html=True)

        input_mode = st.segmented_control(
            "Nguồn hồ sơ",
            ["Hồ sơ có sẵn", "Nhập thông số tùy chỉnh"],
            default="Hồ sơ có sẵn",
            label_visibility="collapsed"
        ) or "Hồ sơ có sẵn"

        if input_mode == "Hồ sơ có sẵn":
            patients_list = get_all_patients()
            p_options = {
                p["patient_id"]: f"{p['patient_id']} — {p['full_name']} (Tuổi {p['age']}, BMI {p['bmi']}, eGFR {p['egfr']})"
                for p in patients_list
            }

            selected_pid = st.selectbox(
                "Hồ sơ bệnh nhân",
                options=list(p_options.keys()),
                format_func=lambda x: p_options[x],
                index=list(p_options.keys()).index(st.session_state.selected_pid) if st.session_state.selected_pid in p_options else 0
            )
            st.session_state.selected_pid = selected_pid
            patient_obj = get_patient(selected_pid)

            p_glu = float(patient_obj.get("fasting_glucose", 0))
            p_hba1c = float(patient_obj.get("hba1c", 0))
            p_egfr = float(patient_obj.get("egfr", 0))
            p_sys = float(patient_obj.get("systolic_bp", 0))
            p_dia = float(patient_obj.get("diastolic_bp", 0))
            p_bmi = float(patient_obj.get("bmi", 0))

            tag_glu_text = "Ngưỡng ĐTĐ" if p_glu >= 126 else ("Tiền ĐTĐ" if p_glu >= 100 else "Bình thường")
            tag_hba1c_text = "Ngưỡng ĐTĐ" if p_hba1c >= 6.5 else ("Tiền ĐTĐ" if p_hba1c >= 5.7 else "Bình thường")
            if p_egfr < 30:
                tag_egfr_text = "Suy thận độ 4-5"
            elif p_egfr < 45:
                tag_egfr_text = "Suy thận độ 3b"
            elif p_egfr < 60:
                tag_egfr_text = "Suy thận độ 3a"
            else:
                tag_egfr_text = "Bình thường"
            tag_bp_text = "Tăng huyết áp" if (p_sys >= 140 or p_dia >= 90) else "Ổn định"
            tag_bmi_text = "Béo phì" if p_bmi >= 30 else ("Thừa cân" if p_bmi >= 25 else "Bình thường")

            with st.container(border=True):
                st.markdown(f"**{patient_obj['full_name']}** (`{patient_obj['patient_id']}`) · {patient_obj['gender']}, {patient_obj['age']} tuổi")
                if patient_obj.get("clinical_notes"):
                    st.caption(patient_obj["clinical_notes"])

                col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
                col_v1.metric("Đường huyết đói", f"{patient_obj['fasting_glucose']} mg/dL", delta=tag_glu_text, delta_color="inverse" if "ĐTĐ" in tag_glu_text else "normal")
                col_v2.metric("HbA1c", f"{patient_obj['hba1c']}%", delta=tag_hba1c_text, delta_color="inverse" if "ĐTĐ" in tag_hba1c_text else "normal")
                col_v3.metric("eGFR", f"{patient_obj['egfr']} mL/phút", delta=tag_egfr_text, delta_color="inverse" if "Suy thận" in tag_egfr_text else "normal")
                col_v4.metric("Huyết áp", f"{patient_obj['systolic_bp']}/{patient_obj['diastolic_bp']} mmHg", delta=tag_bp_text, delta_color="inverse" if "Tăng" in tag_bp_text else "normal")
                col_v5.metric("BMI", f"{patient_obj['bmi']} kg/m²", delta=tag_bmi_text, delta_color="off")

            default_queries = {
                "BN001": "Tôi 58 tuổi, BMI 31.2, HbA1c 7.8%, đường huyết sáng nay là 192 mg/dL và bị suy thận (eGFR = 42 mL/phút). Tôi có dùng được Metformin nguyên liều không và cần ăn uống, tập luyện thế nào?",
                "BN002": "Tôi 46 tuổi, HbA1c 6.1%, glucose đói 115 mg/dL. Tôi có bị đái tháo đường chưa và cần điều chỉnh lối sống ra sao?",
                "BN003": "Tôi 67 tuổi, HbA1c 8.6%, glucose đói 210 mg/dL, suy thận nặng eGFR = 28 mL/phút. Tôi có dùng được Metformin không?"
            }
            user_query_input = st.text_area("Câu hỏi lâm sàng", value=default_queries.get(selected_pid, "Tư vấn phác đồ điều trị."), height=76)
            target_patient_data = patient_obj

        else:
            c_f1, c_f2, c_f3, c_f4 = st.columns(4)
            with c_f1:
                in_pid = st.text_input("Mã BN", value="CUSTOM_01")
                in_age = st.number_input("Tuổi", value=55, min_value=-50, max_value=200)
                in_gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
            with c_f2:
                in_bmi = st.number_input("BMI (kg/m²)", value=28.5, min_value=-10.0, max_value=100.0, step=0.1)
                in_hba1c = st.number_input("HbA1c (%)", value=7.2, min_value=-5.0, max_value=40.0, step=0.1)
            with c_f3:
                in_glu = st.number_input("Glucose đói (mg/dL)", value=165.0, min_value=-100.0, max_value=5000.0, step=1.0)
                in_egfr = st.number_input("eGFR (mL/phút)", value=38.0, min_value=-50.0, max_value=300.0, step=1.0)
            with c_f4:
                in_sys = st.number_input("HA tâm thu (mmHg)", value=140, min_value=-50, max_value=400)
                in_dia = st.number_input("HA tâm trương (mmHg)", value=85, min_value=-50, max_value=300)

            user_query_input = st.text_area("Câu hỏi lâm sàng", value="Tư vấn phác đồ điều trị và chỉnh liều thuốc.", height=76)
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
                "clinical_notes": "Hồ sơ nhập tùy chỉnh"
            }

        if st.button("Chạy hội chẩn", type="primary"):
            with st.spinner("Đang điều phối các tác tử..."):
                consult_res = coordinator.execute_workflow(target_patient_data, user_query=user_query_input)

            if consult_res.get("status") == "error":
                st.error("Dữ liệu đầu vào không hợp lệ. Quy trình đã dừng tại bước xác thực.")
                st.markdown(consult_res["final_report"])
                with st.expander("Vết thực thi", expanded=True):
                    for step in consult_res.get("react_trace", []):
                        st.markdown(f"- **[Bước {step['step']}] {step['agent']}:** {step['observation']}")
            else:
                st.caption(f"Hoàn tất trong `{consult_res['latency_ms']} ms`")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Xác suất ĐTĐ (DNN)", f"{consult_res['dl_output']['risk_analysis']['risk_percentage']}%", delta=consult_res['dl_output']['risk_analysis']['risk_level'], delta_color="inverse" if "CAO" in consult_res['dl_output']['risk_analysis']['risk_level'] else "normal")
                c2.metric("Đỉnh CGM 4h (RNN)", f"{consult_res['dl_output']['cgm_forecast']['peak_glucose']} mg/dL")
                c3.metric("Groundedness", f"{consult_res['groundedness_score']}%")
                c4.metric("Hallucination", f"{consult_res['hallucination_rate']}%")

                st.markdown(consult_res["final_report"])

                with st.expander("Chi tiết quy trình 12 bước", expanded=False):
                    for step in consult_res["react_trace"]:
                        with st.container(border=True):
                            st.markdown(f"**[Bước {step['step']}] {step['phase']} — `{step['agent']}`**")
                            st.markdown(f"- **Thought:** {step['thought']}")
                            st.markdown(f"- **Action:** `{step['action']}`")
                            st.markdown(f"- **Observation:** {step['observation']}")
                            if "message" in step:
                                with st.expander(f"Thông điệp m_i ({step['agent']})"):
                                    st.json(step["message"])

    # =========================================================================
    # VIEW 3: MÔ HÌNH HỌC SÂU & DỰ BÁO CGM
    # =========================================================================
    elif st.session_state.active_view == "dl":
        st.markdown('<div class="view-title">Mô hình Học sâu & Dự báo CGM</div>', unsafe_allow_html=True)
        meta = get_model_metadata()

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Accuracy", f"{meta['evaluation_metrics']['accuracy']*100:.1f}%")
        col_m2.metric("F1-Score", f"{meta['evaluation_metrics']['f1_score']*100:.1f}%")
        col_m3.metric("AUC-ROC", f"{meta['evaluation_metrics']['auc_roc']*100:.1f}%")
        col_m4.metric("Sensitivity", f"{meta['evaluation_metrics']['sensitivity_recall']*100:.1f}%")
        col_m5.metric("Specificity", f"{meta['evaluation_metrics']['specificity']*100:.1f}%")

        with st.expander("Đặc tả kỹ thuật mô hình", expanded=False):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(f"**Phiên bản:** `{meta['model_version']}` · **Tên:** `{meta['model_name']}`")
                st.markdown(f"**DNN 6 tầng:** `{meta['model_architecture']['dnn_layers']}`")
                st.markdown(f"**RNN CGM:** `{meta['model_architecture']['cgm_rnn_architecture']}`")
            with col_d2:
                st.markdown(f"**Dữ liệu huấn luyện:** {meta['training_dataset']['source']} ({meta['training_dataset']['sample_size']:,} mẫu)")
                st.markdown(f"**Tiền xử lý:** {meta['preprocessing_method']['type']} (`{meta['preprocessing_method']['formula']}`)")

        col_arch, col_acts = st.columns([1.15, 1])
        with col_arch:
            st.markdown(r"""
            ##### Cấu trúc các tầng hàm
            | Tầng | Ánh xạ | Biểu thức | Vai trò |
            | :--- | :--- | :--- | :--- |
            | **$f_1$** | $\mathbb{R}^{15} \to \mathbb{R}^{64}$ | $\text{ReLU}(W_1 x + b_1)$ | Mở rộng đặc trưng |
            | **$f_2$** | $\mathbb{R}^{64} \to \mathbb{R}^{64}$ | $\text{ReLU}(W_2 h^{(1)} + b_2)$ | Tương tác phi tuyến |
            | **$f_3$** | $\mathbb{R}^{64} \to \mathbb{R}^{32}$ | $\text{ReLU}(W_3 h^{(2)} + b_3)$ | Nén sơ cấp |
            | **$f_4$** | $\mathbb{R}^{32} \to \mathbb{R}^{16}$ | $\text{ReLU}(W_4 h^{(3)} + b_4)$ | Tinh lọc chuyển hóa |
            | **$f_5$** | $\mathbb{R}^{16} \to \mathbb{R}^{8}$ | $\text{ReLU}(W_5 h^{(4)} + b_5)$ | Tầng Bottleneck |
            | **$f_6$** | $\mathbb{R}^{8} \to [0, 1]$ | $\sigma(W_6 h^{(5)} + b_6)$ | Đầu ra Sigmoid |
            """)

        with col_acts:
            dl_spec = coordinator.dl_agent
            p_curr = get_patient(st.session_state.selected_pid)
            raw_feats = dl_spec.preprocessor.extract_features(p_curr)
            norm_feats = dl_spec.preprocessor.transform(raw_feats)
            y_hat, acts = dl_spec.dnn_model.forward_composition(norm_feats)

            st.markdown(f"##### Bottleneck $h^{{(5)}}$ — {p_curr['full_name']}")
            b_df = pd.DataFrame([acts["h5_bottleneck"]], columns=[f"z_{i+1}" for i in range(8)])
            st.dataframe(b_df.style.background_gradient(cmap="Blues", axis=1), width="stretch")
            st.metric(r"Xác suất $\hat{y} = f_6(h^{(5)})$", f"{y_hat * 100.0:.2f}%")

        st.markdown("---")
        st.markdown("##### Dự báo đường huyết liên tục CGM 4 giờ tới")

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
            name='Lịch sử 12h',
            line=dict(color='#1a73e8', width=2.5),
            marker=dict(size=6, color='#1a73e8')
        ))
        fig_cgm.add_trace(go.Scatter(
            x=[hours_hist[-1]] + hours_fore,
            y=[cgm_hist[-1]] + cgm_pred_res["forecast_4h"],
            mode='lines+markers',
            name='Dự báo RNN 4h',
            line=dict(color='#d93025', width=2.5, dash='dash'),
            marker=dict(size=6, color='#d93025')
        ))
        fig_cgm.add_hrect(
            y0=70, y1=180,
            fillcolor="rgba(30, 142, 62, 0.08)",
            line_width=0,
            annotation_text="Vùng mục tiêu (70–180 mg/dL)",
            annotation_position="top left",
            annotation_font=dict(color="#137333", size=12)
        )
        fig_cgm.add_hline(y=180, line_dash="dot", line_color="#e37400", line_width=1.5)
        fig_cgm.add_hline(y=70, line_dash="dot", line_color="#d93025", line_width=1.5)

        fig_cgm.update_layout(
            title=f"CGM 12h & Dự báo 4h ({p_curr['full_name']})",
            xaxis_title="Thời gian",
            yaxis_title="Glucose (mg/dL)",
            template="plotly_white",
            height=380,
            font=dict(size=12, color="#1f1f1f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cgm, width="stretch")

    # =========================================================================
    # VIEW 4: ĐỒ THỊ TRI THỨC NEO4J
    # =========================================================================
    elif st.session_state.active_view == "kg":
        st.markdown('<div class="view-title">Đồ thị tri thức y khoa</div>', unsafe_allow_html=True)

        g_summary = kg.get_graph_summary()
        if "neo4j_feedback" in st.session_state and st.session_state.neo4j_feedback:
            fb_level, fb_msg = st.session_state.pop("neo4j_feedback")
            if fb_level == "success":
                st.success(fb_msg)
            elif fb_level == "warning":
                st.warning(fb_msg)
            else:
                st.error(fb_msg)

        col_kg1, col_kg2, col_kg3 = st.columns(3)
        col_kg1.metric("Thực thể", g_summary["num_nodes"])
        col_kg2.metric("Quan hệ", g_summary["num_edges"])
        col_kg3.metric("Backend", g_summary.get("backend", "Neo4j"))

        if not g_summary["neo4j_connected"]:
            st.caption(f"Chưa kết nối Neo4j (`{g_summary['neo4j_uri']}`); đang dùng NetworkX dự phòng.")

        with st.expander("Cấu hình kết nối Neo4j", expanded=False):
            neo_col1, neo_col2 = st.columns([2, 1])
            with neo_col1:
                neo_uri = st.text_input("Neo4j Bolt URI", value=kg.neo4j_uri)
                neo_user = st.text_input("Tên đăng nhập", value=kg.neo4j_user)
                neo_pass = st.text_input("Mật khẩu", value=kg.neo4j_password, type="password")
                neo_db = st.text_input("Database", value=kg.neo4j_database or "")
            with neo_col2:
                st.write("")
                st.write("")
                if st.button("Kết nối / Kiểm tra", width="stretch"):
                    ok = kg.connect_neo4j(neo_uri, neo_user, neo_pass, database=neo_db.strip() or None)
                    if ok:
                        kg._ensure_neo4j_seeded("DB_diabet.cypher")
                        st.session_state.neo4j_feedback = ("success", f"Đã kết nối Neo4j tại {neo_uri}.")
                    else:
                        st.session_state.neo4j_feedback = ("error", f"Lỗi kết nối Neo4j ({neo_uri}): {kg.neo4j_error}.")
                    st.rerun()
                if st.button("Nạp DB_diabet.cypher", width="stretch"):
                    ok_cypher = kg.init_neo4j_from_cypher_file("DB_diabet.cypher")
                    if ok_cypher:
                        st.session_state.neo4j_feedback = ("success", "Đã nạp DB_diabet.cypher vào Neo4j.")
                        st.rerun()
                    else:
                        st.warning("Cần kết nối Neo4j trước khi nạp Cypher.")

        graph_options = []
        if g_summary["neo4j_connected"]:
            graph_options.append("Neo4j Live Cypher")
        graph_options.extend([
            "Đồ thị Vinmec (33 nút)",
            "Đồ thị tiêu chuẩn (25 nút)",
        ])
        graph_choice = st.radio("Nguồn đồ thị", graph_options, horizontal=True)

        if "Live Cypher" in graph_choice:
            target_html = "neo4j_live_graph.html"
            kg.generate_pyvis_html(target_html)
        elif "Vinmec" in graph_choice:
            target_html = "kg_graph.html"
        else:
            target_html = "graph_view.html"
            if not os.path.exists(target_html):
                kg.generate_pyvis_html(target_html)

        if os.path.exists(target_html):
            with open(target_html, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.iframe(html_content, height=500)

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            json_file_path = os.path.join("data", "kg_graph.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, "r", encoding="utf-8") as f:
                    st.download_button("Tải kg_graph.json", data=f.read(), file_name="kg_graph.json", mime="application/json", width="stretch")
        with c_dl2:
            if os.path.exists("DB_diabet.cypher"):
                with open("DB_diabet.cypher", "r", encoding="utf-8") as f:
                    st.download_button("Tải DB_diabet.cypher", data=f.read(), file_name="DB_diabet.cypher", mime="text/plain", width="stretch")

        st.markdown("---")
        st.markdown("##### Duyệt đồ thị 2-Hop")
        c_hop1, c_hop2 = st.columns([1, 2])
        with c_hop1:
            node_options = g_summary.get("nodes") or list(kg.G.nodes())
            selected_entity = st.selectbox("Thực thể nguồn", options=node_options, index=0)
        with c_hop2:
            chains = kg.get_multi_hop_facts(selected_entity, max_hops=2)
            st.caption(f"{len(chains)} đường dẫn từ `{selected_entity}`")
            for ch in chains[:6]:
                if ch["hops"] == 2:
                    st.markdown(f"- **2-Hop:** `{ch['source']}` $\\to$ *[{ch['rel1']}]* $\\to$ `{ch['intermediate']}` $\\to$ *[{ch['rel2']}]* $\\to$ `{ch['target']}`")
                else:
                    st.markdown(f"- **1-Hop:** `{ch['source']}` $\\to$ *[{ch['rel1']}]* $\\to$ `{ch['target']}`")

    # =========================================================================
    # VIEW 5: MÔ PHỎNG RL & CSDL
    # =========================================================================
    elif st.session_state.active_view == "rl":
        st.markdown('<div class="view-title">Mô phỏng RL & Cơ sở dữ liệu</div>', unsafe_allow_html=True)

        subtab_rl, subtab_db = st.tabs(["Mô phỏng Q-Learning", "Cơ sở dữ liệu SQLite"])

        with subtab_rl:
            st.caption("Mô phỏng chính sách tuân thủ lối sống trên môi trường tổng hợp; không dùng để kê đơn thực tế.")
            col_qtable, col_compare = st.columns([1, 1])

            with col_qtable:
                st.markdown("##### Bảng Q-Table (4 trạng thái)")
                rl_agent = coordinator.rl_agent.q_agent

                if st.button("Huấn luyện lại (3,000 tập)", type="secondary"):
                    with st.spinner("Đang cập nhật Q-Table..."):
                        from train_models import train_rl_agent
                        new_agent = train_rl_agent(episodes=3000)
                        coordinator.rl_agent.q_agent = new_agent
                        rl_agent = new_agent
                    st.success("Hoàn tất huấn luyện.")

                pol = rl_agent.get_optimal_policy()
                q_data = []
                for s_str, d in pol.items():
                    q_data.append({
                        "Trạng thái": s_str,
                        "Diễn giải": d["state_desc"],
                        "Q(s,0)": round(d["q_0"], 2),
                        "Q(s,1)": round(d["q_1"], 2),
                        "Chính sách": "Can thiệp" if d["best_action"] == 1 else "Duy trì",
                        "Lượt": d["visits"]
                    })
                st.dataframe(pd.DataFrame(q_data), width="stretch")

            with col_compare:
                st.markdown("##### So sánh thích ứng (10,000 bước)")
                exp_data = {
                    "Môi trường": ["Chuẩn (δ = 0.20)", "Thay đổi (δ = 0.10)"],
                    "Q-Learning (RL)": [35.13, 32.07],
                    "Luật tĩnh": [34.75, 30.91]
                }
                df_exp = pd.DataFrame(exp_data)
                fig_bar = px.bar(
                    df_exp,
                    x="Môi trường",
                    y=["Q-Learning (RL)", "Luật tĩnh"],
                    barmode="group",
                    title="Lợi ích kỳ vọng / bước",
                    labels={"value": "Điểm lợi ích", "variable": "Tác tử"},
                    color_discrete_sequence=["#1a73e8", "#bdc1c6"]
                )
                fig_bar.update_layout(
                    template="plotly_white",
                    height=320,
                    font=dict(size=12, color="#1f1f1f"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_bar, width="stretch")

        with subtab_db:
            selected_table = st.radio(
                "Bảng dữ liệu",
                ["agent_action_logs", "patients", "cgm_telemetry_stream"],
                horizontal=True
            )
            conn = get_connection()
            if selected_table == "agent_action_logs":
                df_data = pd.read_sql_query("SELECT * FROM agent_action_logs ORDER BY id DESC LIMIT 50", conn)
            elif selected_table == "patients":
                df_data = pd.read_sql_query("SELECT * FROM patients", conn)
            else:
                df_data = pd.read_sql_query("SELECT * FROM cgm_telemetry_stream ORDER BY id DESC LIMIT 50", conn)
            conn.close()

            st.caption(f"{len(df_data)} bản ghi")
            st.dataframe(df_data, width="stretch")

    # =========================================================================
    # VIEW 6: THƯ VIỆN TÀI LIỆU
    # =========================================================================
    elif st.session_state.active_view == "docs":
        st.markdown('<div class="view-title">Thư viện tài liệu y khoa</div>', unsafe_allow_html=True)

        doc_files = {
            "Quyết định 5481/QĐ-BYT (Bộ Y tế)": "guidelines_byt_5481.txt",
            "Tài liệu chuyên khoa Vinmec": "vinmec_diabetes_article.txt",
            "ADA Standards of Care 2024": "guidelines_ada_2024.txt"
        }
        doc_choice = st.selectbox("Tài liệu", list(doc_files.keys()), label_visibility="collapsed")
        dfname = doc_files[doc_choice]
        dfpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dfname)
        if os.path.exists(dfpath):
            with open(dfpath, "r", encoding="utf-8") as f:
                full_doc_text = f.read()
            st.text_area("Nội dung", value=full_doc_text, height=440, disabled=True, label_visibility="collapsed")
            st.download_button(f"Tải {dfname}", data=full_doc_text, file_name=dfname, mime="text/plain")


# ĐIỂM VÀO CHƯƠNG TRÌNH CHÍNH
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hệ Đa Tác Tử Bệnh Tiểu Đường - PTIT")
    parser.add_argument("--cli", action="store_true", help="Chạy chế độ kiểm thử tự động trên Terminal")
    args, unknown = parser.parse_known_args()

    if args.cli:
        run_cli_mode()
    else:
        run_streamlit_app()
