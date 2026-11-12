"""
=============================================================================
HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG - KHOA CÔNG NGHỆ THÔNG TIN 1
HỌC PHẦN: PHÁT TRIỂN CÁC HỆ THỐNG THÔNG MINH (ISD)
MODULE TRÍCH XUẤT CƠ SỞ TRI THỨC Y KHOA VINMEC SANG ĐỒ THỊ NEO4J & TỆP KG_GRAPH
Tệp thực thi: extract_vinmec_kg.py
Đầu ra:
  1. data/vinmec_diabetes_triples.csv  (Bộ ba quan hệ định dạng CSV)
  2. data/kg_graph.json               (Tệp đồ thị chuẩn Nodes & Edges JSON)
  3. kg_graph.html                    (Tệp HTML đồ thị tương tác PyVis show trực quan)
  4. setup_vinmec_neo4j.cypher        (Script Cypher chạy trực tiếp trên Neo4j Browser)
=============================================================================
"""

import os
import json
import pandas as pd
import networkx as nx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def build_vinmec_knowledge_base():
    """
    Xây dựng danh mục các bộ ba tri thức y khoa chuẩn mực trích xuất từ tài liệu
    Khoa Nội tiết - Bệnh viện Đa khoa Quốc tế Vinmec.
    Bao gồm 6 phân lớp: Disease, Biomarker, Symptom, RiskFactor, Medication, Intervention.
    """
    triples = [
        # --- 1. NHÓM BỆNH HỌC & ĐỊNH NGHĨA (Disease & Etiology) ---
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "caused_by", "rel_label": "Gây ra bởi",
            "object": "Insulin_Resistance", "object_label": "Kháng Insulin mô ngoại vi", "object_type": "Pathology",
            "category": "Etiology", "source": "Vinmec Endocrinology Guide"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "caused_by", "rel_label": "Gây ra bởi",
            "object": "Beta_Cell_Dysfunction", "object_label": "Suy giảm tế bào Beta tụy", "object_type": "Pathology",
            "category": "Etiology", "source": "Vinmec Endocrinology Guide"
        },

        # --- 2. NHÓM CHẨN ĐOÁN & CHỈ SỐ SINH HÓA (Diagnostic Criteria) ---
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "diagnosed_by", "rel_label": "Chẩn đoán xác định bởi",
            "object": "HbA1c_ge_6.5", "object_label": "HbA1c >= 6.5%", "object_type": "Biomarker",
            "category": "Diagnostic", "source": "Vinmec / BYT 5481"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "diagnosed_by", "rel_label": "Chẩn đoán xác định bởi",
            "object": "Fasting_Glucose_ge_126", "object_label": "Đường huyết đói >= 126 mg/dL", "object_type": "Biomarker",
            "category": "Diagnostic", "source": "Vinmec / BYT 5481"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "diagnosed_by", "rel_label": "Chẩn đoán xác định bởi",
            "object": "OGTT_2h_ge_200", "object_label": "Nghiệm pháp OGTT 2h >= 200 mg/dL", "object_type": "Biomarker",
            "category": "Diagnostic", "source": "Vinmec / ADA 2024"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "diagnosed_by", "rel_label": "Chẩn đoán xác định bởi",
            "object": "Random_Glucose_ge_200", "object_label": "Glucose ngẫu nhiên >= 200 mg/dL", "object_type": "Biomarker",
            "category": "Diagnostic", "source": "Vinmec Diagnostic Criteria"
        },

        # --- 3. NHÓM TRIỆU CHỨNG LÂM SÀNG ("HỘI CHỨNG 4 NHIỀU" TỪ VINMEC) ---
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_symptom", "rel_label": "Có triệu chứng",
            "object": "Polyuria", "object_label": "Tiểu nhiều (đặc biệt về đêm)", "object_type": "Symptom",
            "category": "Symptom", "source": "Vinmec 4-P Symptoms"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_symptom", "rel_label": "Có triệu chứng",
            "object": "Polydipsia", "object_label": "Khát nước nhiều liên tục", "object_type": "Symptom",
            "category": "Symptom", "source": "Vinmec 4-P Symptoms"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_symptom", "rel_label": "Có triệu chứng",
            "object": "Polyphagia", "object_label": "Ăn nhiều (mau đói tế bào)", "object_type": "Symptom",
            "category": "Symptom", "source": "Vinmec 4-P Symptoms"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_symptom", "rel_label": "Có triệu chứng",
            "object": "Weight_Loss", "object_label": "Sụt cân nhanh bất thường", "object_type": "Symptom",
            "category": "Symptom", "source": "Vinmec 4-P Symptoms"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_symptom", "rel_label": "Có triệu chứng",
            "object": "Blurred_Vision", "object_label": "Nhìn mờ thể thủy tinh", "object_type": "Symptom",
            "category": "Symptom", "source": "Vinmec Clinical Examination"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_symptom", "rel_label": "Có triệu chứng",
            "object": "Fatigue", "object_label": "Mệt mỏi suy nhược mạn tính", "object_type": "Symptom",
            "category": "Symptom", "source": "Vinmec Clinical Examination"
        },

        # --- 4. NHÓM YẾU TỐ NGUY CƠ (Risk Factors) ---
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_risk_factor", "rel_label": "Có yếu tố nguy cơ",
            "object": "Obesity_BMI_ge_23", "object_label": "Thừa cân/Béo phì (BMI >= 23)", "object_type": "RiskFactor",
            "category": "RiskFactor", "source": "Vinmec Asian Criteria"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_risk_factor", "rel_label": "Có yếu tố nguy cơ",
            "object": "Hypertension", "object_label": "Tăng huyết áp (>=140/90 mmHg)", "object_type": "RiskFactor",
            "category": "RiskFactor", "source": "Vinmec Screening Protocol"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_risk_factor", "rel_label": "Có yếu tố nguy cơ",
            "object": "Family_History", "object_label": "Tiền sử gia đình trực hệ", "object_type": "RiskFactor",
            "category": "RiskFactor", "source": "Vinmec Screening Protocol"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "has_risk_factor", "rel_label": "Có yếu tố nguy cơ",
            "object": "Sedentary_Lifestyle", "object_label": "Lối sống ít vận động", "object_type": "RiskFactor",
            "category": "RiskFactor", "source": "Vinmec Screening Protocol"
        },

        # --- 5. NHÓM BIẾN CHỨNG (Complications) ---
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "causes_complication", "rel_label": "Gây biến chứng",
            "object": "Diabetic_Nephropathy", "object_label": "Bệnh thận đái tháo đường (Suy thận)", "object_type": "Complication",
            "category": "Complication", "source": "Vinmec Complications Report"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "causes_complication", "rel_label": "Gây biến chứng",
            "object": "Diabetic_Retinopathy", "object_label": "Bệnh võng mạc tiểu đường", "object_type": "Complication",
            "category": "Complication", "source": "Vinmec Complications Report"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "causes_complication", "rel_label": "Gây biến chứng",
            "object": "Cardiovascular_Disease", "object_label": "Biến cố tim mạch & Đột quỵ", "object_type": "Complication",
            "category": "Complication", "source": "Vinmec Complications Report"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "causes_complication", "rel_label": "Gây biến chứng",
            "object": "Diabetic_Foot_Ulcer", "object_label": "Loét bàn chân tiểu đường (Nguy cơ đoạn chi)", "object_type": "Complication",
            "category": "Complication", "source": "Vinmec Wound Care Unit"
        },

        # --- 6. NHÓM DƯỢC LÝ & PHÁC ĐỒ ĐIỀU TRỊ (Pharmacology & Treatment) ---
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "treated_by", "rel_label": "Điều trị bước đầu bằng",
            "object": "Metformin_First_Line", "object_label": "Metformin (Glucophage)", "object_type": "Medication",
            "category": "Treatment", "source": "Vinmec Pharmacotherapy"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "treated_by", "rel_label": "Ưu tiên phối hợp",
            "object": "SGLT2_Inhibitor_Empagliflozin", "object_label": "Empagliflozin (Ức chế SGLT2)", "object_type": "Medication",
            "category": "Treatment", "source": "Vinmec Pharmacotherapy"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "treated_by", "rel_label": "Điều trị phối hợp",
            "object": "Sulfonylurea_Gliclazide", "object_label": "Gliclazide (Sulfonylurea)", "object_type": "Medication",
            "category": "Treatment", "source": "Vinmec Pharmacotherapy"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "treated_by", "rel_label": "Liệu pháp tiêm",
            "object": "Insulin_Glargine", "object_label": "Insulin Glargine (Nền)", "object_type": "Medication",
            "category": "Treatment", "source": "Vinmec Pharmacotherapy"
        },

        # --- 7. MẮT XÍCH CHỐNG CHỈ ĐỊNH VÀ AN TOÀN 2-HOP (Cốt tử trong suy luận) ---
        {
            "subject": "Metformin_First_Line", "subject_label": "Metformin (Glucophage)", "subject_type": "Medication",
            "relation": "dose_reduction_when", "rel_label": "Giảm 50% liều khi",
            "object": "eGFR_30_to_45_mL_min", "object_label": "eGFR 30 - 45 mL/phút (Suy thận vừa)", "object_type": "Biomarker",
            "category": "SafetyRule", "source": "Vinmec Safety Alert"
        },
        {
            "subject": "Metformin_First_Line", "subject_label": "Metformin (Glucophage)", "subject_type": "Medication",
            "relation": "contraindicated_with", "rel_label": "Chống chỉ định tuyệt đối với",
            "object": "Severe_Renal_Impairment_eGFR_lt_30", "object_label": "eGFR < 30 mL/phút (Suy thận nặng)", "object_type": "Biomarker",
            "category": "SafetyRule", "source": "Vinmec / BYT 5481 Alert"
        },
        {
            "subject": "SGLT2_Inhibitor_Empagliflozin", "subject_label": "Empagliflozin (Ức chế SGLT2)", "subject_type": "Medication",
            "relation": "proven_benefit", "rel_label": "Lợi ích đã chứng minh",
            "object": "Renal_and_Heart_Failure_Protection", "object_label": "Bảo vệ thận và giảm suy tim", "object_type": "Intervention",
            "category": "ClinicalBenefit", "source": "Vinmec Cardio-Renal Evidence"
        },
        {
            "subject": "Hypertension", "subject_label": "Tăng huyết áp", "subject_type": "Disease",
            "relation": "treated_by", "rel_label": "Điều trị ưu tiên bằng",
            "object": "ACE_Inhibitor", "object_label": "Thuốc ức chế men chuyển (ACEi)", "object_type": "Medication",
            "category": "Treatment", "source": "Vinmec Hypertension Protocol"
        },
        {
            "subject": "ACE_Inhibitor", "subject_label": "Thuốc ức chế men chuyển (ACEi)", "subject_type": "Medication",
            "relation": "proven_benefit", "rel_label": "Lợi ích đã chứng minh",
            "object": "Renal_Hemodynamic_Protection", "object_label": "Bảo vệ huyết động màng lọc cầu thận", "object_type": "Intervention",
            "category": "ClinicalBenefit", "source": "Vinmec Nephrology Guidelines"
        },
        {
            "subject": "Severe_Renal_Impairment_eGFR_lt_30", "subject_label": "eGFR < 30 mL/phút", "subject_type": "Biomarker",
            "relation": "requires", "rel_label": "Bắt buộc chuyển sang",
            "object": "Insulin_Therapy", "object_label": "Phác đồ Insulin kiểm soát tích cực", "object_type": "Intervention",
            "category": "ClinicalAction", "source": "Vinmec Clinical Safety"
        },

        # --- 8. PHÁC ĐỒ DINH DƯỠNG & LỐI SỐNG VINMEC ---
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "managed_by", "rel_label": "Quản lý lối sống bằng",
            "object": "Low_GI_Diet_Plan", "object_label": "Chế độ ăn giảm 35% tinh bột GI cao", "object_type": "Intervention",
            "category": "Lifestyle", "source": "Vinmec Nutrition Department"
        },
        {
            "subject": "Diabetes_Type_2", "subject_label": "Đái tháo đường Típ 2", "subject_type": "Disease",
            "relation": "managed_by", "rel_label": "Quản lý lối sống bằng",
            "object": "Brisk_Walking_30min", "object_label": "Đi bộ nhanh 30 phút sau bữa ăn chính", "object_type": "Intervention",
            "category": "Lifestyle", "source": "Vinmec Rehabilitation"
        }
    ]
    return triples


def export_all_formats():
    print("=" * 75)
    print("TRÍCH XUẤT CƠ SỞ TRI THỨC Y KHOA VINMEC CHO BỆNH TIỂU ĐƯỜNG")
    print("=" * 75)

    triples = build_vinmec_knowledge_base()
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. XUẤT FILE CSV: data/vinmec_diabetes_triples.csv
    csv_path = os.path.join(DATA_DIR, "vinmec_diabetes_triples.csv")
    df_triples = pd.DataFrame(triples)
    df_triples.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[1] Đã tạo tệp CSV thành công: {csv_path} ({len(df_triples)} bộ ba tri thức)")

    # 2. XÂY DỰNG ĐỒ THỊ NETWORKX VÀ XUẤT FILE JSON: data/kg_graph.json
    G = nx.DiGraph()
    nodes_dict = {}

    for t in triples:
        s_id = t["subject"]
        o_id = t["object"]

        if s_id not in nodes_dict:
            nodes_dict[s_id] = {
                "id": s_id,
                "label": t["subject_label"],
                "type": t["subject_type"],
                "category": t["category"]
            }
        if o_id not in nodes_dict:
            nodes_dict[o_id] = {
                "id": o_id,
                "label": t["object_label"],
                "type": t["object_type"],
                "category": t["category"]
            }

        G.add_edge(s_id, o_id, relation=t["relation"], label=t["rel_label"], category=t["category"], source=t["source"])

    # Tạo cấu trúc node-link JSON tiêu chuẩn
    json_nodes = []
    color_map = {
        "Disease": "#FF4B4B",         # Đỏ đậm
        "Biomarker": "#00D2D3",       # Xanh ngọc
        "Symptom": "#FFA801",         # Cam tươi
        "RiskFactor": "#FF6B6B",      # Hồng san hô
        "Medication": "#2E86DE",      # Xanh biển
        "Intervention": "#10AC84",    # Xanh lục bảo
        "Pathology": "#54A0FF"        # Xanh lơ
    }

    for n_id, meta in nodes_dict.items():
        node_type = meta["type"]
        color = color_map.get(node_type, "#A4B0BE")
        json_nodes.append({
            "id": n_id,
            "label": meta["label"],
            "group": node_type,
            "category": meta["category"],
            "color": color,
            "size": 28 if node_type == "Disease" else (22 if node_type in ["Medication", "Biomarker"] else 18)
        })

    json_links = []
    for u, v, data in G.edges(data=True):
        json_links.append({
            "source": u,
            "target": v,
            "relation": data.get("relation", ""),
            "label": data.get("label", ""),
            "category": data.get("category", ""),
            "source_doc": data.get("source", "")
        })

    kg_graph_data = {
        "metadata": {
            "title": "Clinical Diabetes Knowledge Graph (Vinmec Medical Source)",
            "version": "2.0",
            "hospital_source": "Bệnh viện Đa khoa Quốc tế Vinmec - Khoa Nội tiết",
            "num_nodes": len(json_nodes),
            "num_edges": len(json_links)
        },
        "nodes": json_nodes,
        "links": json_links
    }

    json_path = os.path.join(DATA_DIR, "kg_graph.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(kg_graph_data, f, ensure_ascii=False, indent=2)
    print(f"[2] Đã tạo tệp KG_GRAPH JSON chuẩn: {json_path} ({len(json_nodes)} nodes, {len(json_links)} links)")

    # 3. TẠO FILE HTML TRỰC QUAN HÓA TƯƠNG TÁC: kg_graph.html
    html_path = os.path.join(BASE_DIR, "kg_graph.html")
    try:
        from pyvis.network import Network
        net = Network(height="750px", width="100%", bgcolor="#0F172A", font_color="#F8FAFC", directed=True)
        
        for n in json_nodes:
            title_hover = f"<b>{n['label']}</b><br>ID: {n['id']}<br>Phân loại: {n['group']}"
            net.add_node(n["id"], label=n["label"], color=n["color"], size=n["size"], title=title_hover, shape="dot")

        for l in json_links:
            edge_hover = f"Quan hệ: {l['label']} ({l['relation']})<br>Nguồn: {l['source_doc']}"
            net.add_edge(l["source"], l["target"], title=edge_hover, label=l["label"], color="#64748B", arrows="to", font={"size": 11, "color": "#94A3B8"})

        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -5000,
              "centralGravity": 0.25,
              "springLength": 110,
              "springConstant": 0.04
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)
        net.save_graph(html_path)
        print(f"[3] Đã tạo tệp HTML trực quan hóa đồ thị: {html_path}")
    except Exception as e:
        print(f"[!] Lỗi khi xuất pyvis: {e}")

    # 4. Ghi chú tệp Cypher chính thức của dự án: DB_diabet.cypher
    cypher_path = os.path.join(BASE_DIR, "DB_diabet.cypher")
    print(f"[4] Tệp Cypher tổng hợp chính thức của dự án: {cypher_path}")
    print("\n" + "=" * 75)
    print("HOÀN TẤT QUÁ TRÌNH TRÍCH XUẤT ĐỒ THỊ TRI THỨC.")
    print("=" * 75)

if __name__ == "__main__":
    export_all_formats()

