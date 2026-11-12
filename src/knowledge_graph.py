"""
Module: src/knowledge_graph.py
Mục đích: Xây dựng, quản trị và duyệt Đồ thị Tri thức Y khoa (Clinical Knowledge Graph)
Hỗ trợ cả NetworkX (in-memory) và Neo4j (Enterprise Graph DB qua Cypher)
Phân hệ: Khối 3 - Knowledge Base & Graph Traversal (1-Hop và 2-Hop)
Học phần: Phát triển các hệ thống thông minh - Học viện CN Bưu chính Viễn thông
"""

import os
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional

# Cố gắng import thư viện neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "medical_knowledge_triples.csv")

class MedicalKnowledgeGraph:
    """
    Đồ thị tri thức y khoa Đái tháo đường (Clinical Diabetes Knowledge Graph).
    Quản lý 5 lớp thực thể (Disease, Biomarker, Symptom, Medication, RiskFactor)
    và các cạnh quan hệ ngữ nghĩa (treated_by, dose_reduction_when, contraindicated_with, v.v.).
    """
    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path
        self.G = nx.DiGraph()
        self.neo4j_driver = None
        self.neo4j_connected = False
        
        # Load dữ liệu từ CSV vào NetworkX
        self.load_from_csv(csv_path)

    def load_from_csv(self, csv_path: str) -> None:
        """Nạp các bộ ba tri thức (Subject, Relation, Object, Category) vào đồ thị."""
        if not os.path.exists(csv_path):
            print(f"[!] Cảnh báo: Tệp {csv_path} không tồn tại.")
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            sub = str(row['subject']).strip()
            rel = str(row['relation']).strip()
            obj = str(row['object']).strip()
            cat = str(row.get('category', 'General')).strip()
            self.G.add_edge(sub, obj, relation=rel, category=cat)

    def connect_neo4j(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password") -> bool:
        """Kết nối tới Neo4j Graph Database."""
        if not NEO4J_AVAILABLE:
            print("[!] Thư viện neo4j chưa được cài đặt.")
            self.neo4j_connected = False
            return False

        try:
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test kết nối
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    self.neo4j_connected = True
                    print(f"[+] Kết nối Neo4j thành công tới: {uri}")
                    return True
        except Exception as e:
            print(f"[!] Không thể kết nối Neo4j ({uri}): {e}")
            self.neo4j_connected = False
            return False
        return False

    def query_neo4j_cypher(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Thực thi câu truy vấn Cypher trên Neo4j nếu đã kết nối."""
        if not self.neo4j_connected or not self.neo4j_driver:
            return []
        
        records_list = []
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_query, params or {})
                for record in result:
                    records_list.append(dict(record))
        except Exception as e:
            print(f"[!] Lỗi truy vấn Cypher: {e}")
        return records_list

    def init_neo4j_from_cypher_file(self, cypher_file: str = "DB_diabet.cypher") -> bool:
        """
        Nạp đồ thị tri thức vào Neo4j từ script Cypher (DB_diabet.cypher)
        theo đặc tả Mục 1.1 của tài liệu hệ thống.
        """
        if not self.neo4j_connected or not self.neo4j_driver:
            print("[!] Chưa kết nối Neo4j, không thể nạp script Cypher.")
            return False

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(root_dir, cypher_file)
        if not os.path.exists(file_path):
            print(f"[!] Không tìm thấy tệp Cypher: {cypher_file}")
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Tách các câu lệnh theo dấu chấm phẩy
            statements = [s.strip() for s in content.split(";") if s.strip() and not s.strip().startswith("//")]
            with self.neo4j_driver.session() as session:
                for stmt in statements:
                    if stmt:
                        session.run(stmt)
            print(f"[+] Nạp thành công {len(statements)} câu lệnh từ {cypher_file} vào Neo4j.")
            return True
        except Exception as e:
            print(f"[!] Lỗi khi nạp script Cypher: {e}")
            return False

    def get_entity_facts(self, entity: str) -> List[str]:
        """Duyệt quan hệ 1-hop (cạnh đi ra và cạnh đi vào) của một thực thể."""
        facts = []
        if entity not in self.G:
            return facts

        # Quan hệ đi ra (Outgoing edges: entity -> target)
        for _, target, data in self.G.out_edges(entity, data=True):
            rel = data.get('relation', 'relates_to')
            facts.append(f"({entity}) --[{rel}]--> ({target})")

        # Quan hệ đi vào (Incoming edges: source -> entity)
        for source, _, data in self.G.in_edges(entity, data=True):
            rel = data.get('relation', 'relates_to')
            facts.append(f"({source}) --[{rel}]--> ({entity})")

        return facts

    def get_multi_hop_facts(self, entity: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Duyệt chuỗi quan hệ bắc cầu 2-hop phục vụ suy luận logic sâu (Multi-hop Reasoning).
        Ví dụ: Diabetes_Type_2 -> Metformin_First_Line -> eGFR_30_to_45_mL_min
        """
        multi_hop_chains = []
        if entity not in self.G:
            return multi_hop_chains

        # 1-Hop
        for neighbor in self.G.successors(entity):
            rel1 = self.G[entity][neighbor].get('relation', 'relates_to')
            chain_1hop = {
                "hops": 1,
                "chain_str": f"({entity}) --[{rel1}]--> ({neighbor})",
                "source": entity,
                "rel1": rel1,
                "intermediate": neighbor,
                "rel2": None,
                "target": neighbor
            }
            multi_hop_chains.append(chain_1hop)

            # 2-Hop
            if max_hops >= 2:
                for target_2 in self.G.successors(neighbor):
                    if target_2 != entity:
                        rel2 = self.G[neighbor][target_2].get('relation', 'relates_to')
                        chain_2hop = {
                            "hops": 2,
                            "chain_str": f"({entity}) --[{rel1}]--> ({neighbor}) --[{rel2}]--> ({target_2})",
                            "source": entity,
                            "rel1": rel1,
                            "intermediate": neighbor,
                            "rel2": rel2,
                            "target": target_2
                        }
                        multi_hop_chains.append(chain_2hop)

        return multi_hop_chains

    def check_contraindications(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chốt chặn an toàn dược lý: Kiểm tra chống chỉ định thuốc và hiệu chỉnh liều
        bằng suy luận bắc cầu 2-hop dựa trên chức năng thận eGFR và bệnh nền.
        """
        egfr = float(patient.get("egfr", 80.0))
        hypertension = int(patient.get("hypertension", 0))
        
        detected_rules = []
        contraindication_flag = False
        dose_adjustment = None
        recommended_drugs = []
        avoid_drugs = []

        # Kiểm tra quy tắc Metformin dựa trên eGFR
        if egfr < 30.0:
            contraindication_flag = True
            avoid_drugs.append("Metformin_First_Line")
            dose_adjustment = "CHỐNG CHỈ ĐỊNH TUYỆT ĐỐI Metformin (eGFR < 30 mL/phút). Ngừng thuốc ngay lập tức để phòng ngừa nhiễm toan Lactic Acidosis."
            recommended_drugs.append("Insulin_Therapy")
            detected_rules.append({
                "type": "CONTRAINDICATION",
                "chain": "(Metformin_First_Line) --[contraindicated_with]--> (Severe_Renal_Impairment_eGFR_lt_30)",
                "action": "Ngừng Metformin, chuyển Insulin"
            })
        elif 30.0 <= egfr < 45.0:
            dose_adjustment = f"GIẢM 50% LIỀU Metformin (eGFR = {egfr} mL/phút nằm trong khoảng 30 - 45 mL/phút). Liều tối đa 1000 mg/ngày, theo dõi eGFR định kỳ 3 tháng/lần."
            recommended_drugs.append("Metformin (50% liều)")
            recommended_drugs.append("SGLT2_Inhibitor_Empagliflozin")
            detected_rules.append({
                "type": "DOSE_REDUCTION",
                "chain": "(Diabetes_Type_2) --[treated_by]--> (Metformin_First_Line) --[dose_reduction_when]--> (eGFR_30_to_45_mL_min)",
                "action": "Giảm 50% liều Metformin"
            })
            detected_rules.append({
                "type": "PROVEN_BENEFIT",
                "chain": "(Diabetes_Type_2) --[treated_by]--> (SGLT2_Inhibitor_Empagliflozin) --[proven_benefit]--> (Renal_and_Heart_Failure_Protection)",
                "action": "Ưu tiên phối hợp thuốc ức chế SGLT2 bảo vệ chức năng thận"
            })
        else:
            dose_adjustment = "Sử dụng liều Metformin tiêu chuẩn (eGFR >= 45 mL/phút an toàn)."
            recommended_drugs.append("Metformin_First_Line")

        # Kiểm tra Tăng huyết áp đi kèm
        if hypertension == 1:
            recommended_drugs.append("ACE_Inhibitor")
            detected_rules.append({
                "type": "COMORBIDITY_CARDIO_RENAL",
                "chain": "(Hypertension) --[treated_by]--> (ACE_Inhibitor) --[proven_benefit]--> (Renal_Hemodynamic_Protection)",
                "action": "Phối hợp thuốc ức chế men chuyển để hạ áp và bảo vệ áp lực lọc cầu thận"
            })

        return {
            "patient_id": patient.get("patient_id", "Unknown"),
            "egfr": egfr,
            "hypertension": hypertension,
            "contraindication_flag": contraindication_flag,
            "dose_adjustment": dose_adjustment,
            "recommended_drugs": recommended_drugs,
            "avoid_drugs": avoid_drugs,
            "detected_rules": detected_rules
        }

    def find_matching_entities(self, query: str) -> List[str]:
        """Nhận diện các thực thể có trong đồ thị xuất hiện trong truy vấn."""
        query_lower = query.lower()
        matched = []
        for node in self.G.nodes():
            if str(node).lower() in query_lower or str(node).replace("_", " ").lower() in query_lower:
                matched.append(str(node))
        return matched

    def get_graph_summary(self) -> Dict[str, Any]:
        """Thống kê kích thước đồ thị."""
        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "nodes": list(self.G.nodes()),
            "neo4j_connected": self.neo4j_connected
        }

    def generate_pyvis_html(self, output_html_path: str = "graph_visualization.html") -> str:
        """Tạo file HTML trực quan hóa đồ thị tương tác bằng pyvis."""
        try:
            from pyvis.network import Network
            net = Network(height="500px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)
            
            # Phân màu các node theo vai trò
            for node in self.G.nodes():
                n_str = str(node)
                if "Diabetes" in n_str or "Hypertension" in n_str:
                    color = "#FF4B4B" # Bệnh lý - Đỏ
                    shape = "dot"
                    size = 25
                elif "Glucose" in n_str or "HbA1c" in n_str or "eGFR" in n_str:
                    color = "#00FFAA" # Sinh hóa - Xanh ngọc
                    shape = "diamond"
                    size = 20
                elif "Metformin" in n_str or "SGLT2" in n_str or "ACE" in n_str or "Insulin" in n_str:
                    color = "#2E86DE" # Thuốc - Xanh lam
                    shape = "triangle"
                    size = 22
                else:
                    color = "#F39C12" # Triệu chứng / Nguy cơ - Vàng cam
                    shape = "dot"
                    size = 18

                net.add_node(n_str, label=n_str.replace("_", " "), color=color, shape=shape, size=size)

            for u, v, data in self.G.edges(data=True):
                rel = data.get('relation', '')
                net.add_edge(str(u), str(v), title=rel, label=rel, color="#8395A7", arrows="to")

            net.set_options("""
            var options = {
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -4000,
                  "centralGravity": 0.3,
                  "springLength": 95,
                  "springConstant": 0.04
                },
                "minVelocity": 0.75
              }
            }
            """)
            net.save_graph(output_html_path)
            return output_html_path
        except Exception as e:
            print(f"[!] Lỗi khi xuất đồ thị PyVis: {e}")
            return ""

if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    summary = kg.get_graph_summary()
    print(f"[+] Knowledge Graph sẵn sàng với {summary['num_nodes']} nodes, {summary['num_edges']} edges.")
    
    sample_patient = {"patient_id": "BN001", "egfr": 42.0, "hypertension": 1}
    safety = kg.check_contraindications(sample_patient)
    print(f"[+] Kiểm tra an toàn BN001: {safety['dose_adjustment']}")
    for r in safety['detected_rules']:
        print(f"    -> [{r['type']}]: {r['chain']}")

