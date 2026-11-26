"""
Module: src/knowledge_graph.py
Mục đích: Xây dựng, quản trị và duyệt Đồ thị Tri thức Y khoa (Clinical Knowledge Graph)
Mặc định kết nối Neo4j (Enterprise Graph DB qua Cypher - DB_diabet.cypher),
tự động fallback sang NetworkX (in-memory) khi chưa khởi động Neo4j server.
Phân hệ: Khối 3 - Knowledge Base & Graph Traversal (1-Hop và 2-Hop)
Học phần: Phát triển các hệ thống thông minh - Học viện CN Bưu chính Viễn thông
"""

import os
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Optional

# Cố gắng import thư viện neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Fix SSL cho AuraDB trên Python 3.14+: trỏ SSL_CERT_FILE tới certifi CA bundle
# để tránh lỗi "certificate verify failed: self-signed certificate in chain"
try:
    import certifi as _certifi
    import os as _os
    if not _os.environ.get("SSL_CERT_FILE"):
        _os.environ["SSL_CERT_FILE"] = _certifi.where()
    if not _os.environ.get("REQUESTS_CA_BUNDLE"):
        _os.environ["REQUESTS_CA_BUNDLE"] = _certifi.where()
    del _certifi, _os
except ImportError:
    pass  # certifi chưa cài — vẫn thử kết nối bình thường

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, "data", "medical_knowledge_triples.csv")
DEFAULT_ENV_PATH = os.path.join(ROOT_DIR, ".env")

DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password"

# Ưu tiên tên quan hệ chuẩn tiếng Anh khi đồ thị có song song quan hệ EN & VI
_PREFERRED_REL_ORDER = {
    "DIAGNOSED_BY": 1,
    "TREATED_BY": 2,
    "DOSE_REDUCTION_WHEN": 3,
    "CONTRAINDICATED_WITH": 4,
    "REQUIRES": 5,
    "PROVEN_BENEFIT": 6,
    "HAS_SYMPTOM": 7,
    "HAS_RISK_FACTOR": 8,
    "COMORBID_WITH": 9,
    "CAUSES_COMPLICATION": 10,
    "MANAGED_BY": 11,
}


def _load_dotenv_file(env_path: Optional[str] = None) -> Dict[str, str]:
    """Đọc cấu hình key=value từ tệp .env (nếu có) mà không phụ thuộc thư viện ngoài."""
    target_path = env_path or DEFAULT_ENV_PATH
    values: Dict[str, str] = {}
    if not os.path.exists(target_path):
        return values

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                if key:
                    values[key] = val
    except OSError:
        pass
    return values


def resolve_neo4j_config(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Tổng hợp cấu hình Neo4j theo thứ tự ưu tiên:
    Tham số truyền trực tiếp -> Biến môi trường -> Tệp .env -> Giá trị mặc định.
    """
    dotenv_vals = _load_dotenv_file()

    def _pick(explicit: Optional[str], env_key: str, default_val: Optional[str]) -> Optional[str]:
        if explicit is not None and str(explicit).strip() != "":
            return str(explicit).strip()
        env_val = os.environ.get(env_key)
        if env_val is not None and env_val.strip() != "":
            return env_val.strip()
        dot_val = dotenv_vals.get(env_key)
        if dot_val is not None and dot_val.strip() != "":
            return dot_val.strip()
        return default_val

    # Hỗ trợ cả NEO4J_USER (chuẩn) và NEO4J_USERNAME (AuraDB credentials file)
    resolved_user = _pick(user, "NEO4J_USER", None) or _pick(None, "NEO4J_USERNAME", DEFAULT_NEO4J_USER)

    return {
        "uri": _pick(uri, "NEO4J_URI", DEFAULT_NEO4J_URI),
        "user": resolved_user,
        "password": _pick(password, "NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD),
        "database": _pick(database, "NEO4J_DATABASE", None),
    }


def split_cypher_statements(content: str) -> List[str]:
    """
    Tách script Cypher thành các câu lệnh độc lập:
    - Bỏ qua comment dòng (//...) và comment khối (/*...*/) ngoài chuỗi.
    - Không tách nhầm dấu ';' nằm bên trong chuỗi ký tự trích dẫn.
    """
    statements: List[str] = []
    buf: List[str] = []
    n = len(content)
    i = 0
    in_single = False
    in_double = False
    escape = False

    while i < n:
        ch = content[i]

        if in_single or in_double:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif in_single and ch == "'":
                in_single = False
            elif in_double and ch == '"':
                in_double = False
            i += 1
            continue

        # Bỏ qua comment dạng // ...
        if ch == "/" and i + 1 < n and content[i + 1] == "/":
            while i < n and content[i] != "\n":
                i += 1
            continue

        # Bỏ qua comment dạng /* ... */
        if ch == "/" and i + 1 < n and content[i + 1] == "*":
            i += 2
            while i + 1 < n and not (content[i] == "*" and content[i + 1] == "/"):
                i += 1
            i += 2
            continue

        if ch == "'":
            in_single = True
            buf.append(ch)
        elif ch == '"':
            in_double = True
            buf.append(ch)
        elif ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)

        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)

    return statements


def _pick_preferred_rel(rels: List[str]) -> str:
    """Chọn tên quan hệ đại diện nhất khi tồn tại cặp cạnh song ngữ EN/VI."""
    if not rels:
        return "RELATES_TO"
    return min(rels, key=lambda r: (_PREFERRED_REL_ORDER.get(r, 99 if r.isascii() else 199), r))


class MedicalKnowledgeGraph:
    """
    Đồ thị tri thức y khoa Đái tháo đường (Clinical Diabetes Knowledge Graph).
    Mặc định tự động kết nối Neo4j (nạp DB_diabet.cypher nếu CSDL rỗng) và
    lưu sẵn đồ thị NetworkX từ CSV làm lớp dự phòng (fallback).
    """

    def __init__(
        self,
        csv_path: str = CSV_PATH,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
        auto_connect_neo4j: bool = True,
        auto_init_cypher: bool = True,
        cypher_file: str = "DB_diabet.cypher",
    ):
        self.csv_path = csv_path
        self.G = nx.DiGraph()
        self.neo4j_driver = None
        self.neo4j_connected = False
        self.neo4j_error: Optional[str] = None

        cfg = resolve_neo4j_config(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
        self.neo4j_uri: str = cfg["uri"] or DEFAULT_NEO4J_URI
        self.neo4j_user: str = cfg["user"] or DEFAULT_NEO4J_USER
        self.neo4j_password: str = cfg["password"] or DEFAULT_NEO4J_PASSWORD
        self.neo4j_database: Optional[str] = cfg["database"]

        # Luôn nạp trước dữ liệu CSV vào NetworkX để dự phòng khi Neo4j chưa chạy
        self.load_from_csv(csv_path)

        if auto_connect_neo4j:
            connected = self.connect_neo4j(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
                database=self.neo4j_database,
            )
            if connected and auto_init_cypher:
                self._ensure_neo4j_seeded(cypher_file)

    def _open_session(self):
        if not self.neo4j_driver:
            raise RuntimeError("Neo4j driver chưa được khởi tạo.")
        if self.neo4j_database:
            return self.neo4j_driver.session(database=self.neo4j_database)
        return self.neo4j_driver.session()

    def close(self) -> None:
        """Đóng kết nối Neo4j driver hiện tại (nếu có)."""
        if self.neo4j_driver is not None:
            try:
                self.neo4j_driver.close()
            except Exception:
                pass
            self.neo4j_driver = None
        self.neo4j_connected = False

    def load_from_csv(self, csv_path: str) -> None:
        """Nạp các bộ ba tri thức (Subject, Relation, Object, Category) vào đồ thị NetworkX."""
        if not os.path.exists(csv_path):
            print(f"[!] Cảnh báo: Tệp {csv_path} không tồn tại.")
            return

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            sub = str(row["subject"]).strip()
            rel = str(row["relation"]).strip()
            obj = str(row["object"]).strip()
            cat = str(row.get("category", "General")).strip()
            self.G.add_edge(sub, obj, relation=rel, category=cat)

    def connect_neo4j(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        timeout_seconds: float = 3.0,
    ) -> bool:
        """Kết nối tới Neo4j Graph Database (mặc định dùng cấu hình đã phân giải)."""
        target_uri = uri or self.neo4j_uri
        target_user = user or self.neo4j_user
        target_password = password if password is not None else self.neo4j_password
        target_db = database if database is not None else self.neo4j_database

        self.neo4j_uri = target_uri
        self.neo4j_user = target_user
        self.neo4j_password = target_password
        self.neo4j_database = target_db

        if not NEO4J_AVAILABLE:
            self.neo4j_error = "Thư viện neo4j chưa được cài đặt."
            print(f"[!] {self.neo4j_error}")
            self.neo4j_connected = False
            return False

        self.close()
        try:
            self.neo4j_driver = GraphDatabase.driver(
                target_uri,
                auth=(target_user, target_password),
                connection_timeout=timeout_seconds,
            )
            self.neo4j_driver.verify_connectivity()
            with self._open_session() as session:
                record = session.run("RETURN 1 AS test").single()
                if record and record["test"] == 1:
                    self.neo4j_connected = True
                    self.neo4j_error = None
                    print(f"[+] Kết nối Neo4j thành công tới: {target_uri}")
                    return True
        except Exception as e:
            self.neo4j_error = str(e)
            print(f"[!] Không thể kết nối Neo4j ({target_uri}): {e}")
            self.close()
            return False
        return False

    def _ensure_neo4j_seeded(self, cypher_file: str = "DB_diabet.cypher") -> None:
        """Tự động nạp DB_diabet.cypher nếu Neo4j đang trống dữ liệu DiabetesConcept."""
        rows = self.query_neo4j_cypher("MATCH (n:DiabetesConcept) RETURN count(n) AS cnt")
        if rows and int(rows[0].get("cnt", 0)) == 0:
            print(f"[INFO] Neo4j đang trống, tự động nạp tri thức từ {cypher_file}...")
            self.init_neo4j_from_cypher_file(cypher_file)

    def query_neo4j_cypher(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Thực thi câu truy vấn Cypher trên Neo4j nếu đã kết nối."""
        if not self.neo4j_connected or not self.neo4j_driver:
            return []

        records_list: List[Dict[str, Any]] = []
        try:
            with self._open_session() as session:
                result = session.run(cypher_query, params or {})
                for record in result:
                    records_list.append(dict(record))
            self.neo4j_error = None
        except Exception as e:
            self.neo4j_error = str(e)
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

        file_path = cypher_file if os.path.isabs(cypher_file) else os.path.join(ROOT_DIR, cypher_file)
        if not os.path.exists(file_path):
            print(f"[!] Không tìm thấy tệp Cypher: {cypher_file}")
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            statements = split_cypher_statements(content)
            with self._open_session() as session:
                for stmt in statements:
                    session.run(stmt)
            print(f"[+] Nạp thành công {len(statements)} câu lệnh từ {cypher_file} vào Neo4j.")
            return True
        except Exception as e:
            self.neo4j_error = str(e)
            print(f"[!] Lỗi khi nạp script Cypher: {e}")
            return False

    def get_entity_facts(self, entity: str) -> List[str]:
        """Duyệt quan hệ 1-hop (cạnh đi ra và cạnh đi vào) của một thực thể."""
        if self.neo4j_connected:
            facts: List[str] = []
            out_rows = self.query_neo4j_cypher(
                """
                MATCH (s {name: $entity})-[r]->(t)
                WHERE t.name IS NOT NULL
                RETURN s.name AS src, t.name AS dst, collect(DISTINCT type(r)) AS rels
                ORDER BY dst
                """,
                {"entity": entity},
            )
            for row in out_rows:
                rel = _pick_preferred_rel(row.get("rels", []))
                facts.append(f"({row['src']}) --[{rel}]--> ({row['dst']})")

            in_rows = self.query_neo4j_cypher(
                """
                MATCH (s)-[r]->(t {name: $entity})
                WHERE s.name IS NOT NULL
                RETURN s.name AS src, t.name AS dst, collect(DISTINCT type(r)) AS rels
                ORDER BY src
                """,
                {"entity": entity},
            )
            for row in in_rows:
                rel = _pick_preferred_rel(row.get("rels", []))
                facts.append(f"({row['src']}) --[{rel}]--> ({row['dst']})")

            if facts or self.neo4j_error is None:
                return facts

        facts = []
        if entity not in self.G:
            return facts

        for _, target, data in self.G.out_edges(entity, data=True):
            rel = data.get("relation", "relates_to")
            facts.append(f"({entity}) --[{rel}]--> ({target})")

        for source, _, data in self.G.in_edges(entity, data=True):
            rel = data.get("relation", "relates_to")
            facts.append(f"({source}) --[{rel}]--> ({entity})")

        return facts

    def get_multi_hop_facts(self, entity: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Duyệt chuỗi quan hệ bắc cầu 1-hop và 2-hop phục vụ suy luận logic sâu (Multi-hop Reasoning).
        """
        if self.neo4j_connected:
            hop1_rows = self.query_neo4j_cypher(
                """
                MATCH (a {name: $entity})-[r1]->(b)
                WHERE b.name IS NOT NULL
                OPTIONAL MATCH (b)-[r2]->(c)
                WHERE c.name IS NOT NULL AND c.name <> a.name
                RETURN a.name AS src, b.name AS mid,
                       collect(DISTINCT type(r1)) AS rels1,
                       count(DISTINCT c) AS two_hop_cnt
                ORDER BY two_hop_cnt DESC, mid ASC
                """,
                {"entity": entity},
            )

            hop2_by_mid: Dict[str, List[Dict[str, Any]]] = {}
            if max_hops >= 2 and hop1_rows:
                hop2_rows = self.query_neo4j_cypher(
                    """
                    MATCH (a {name: $entity})-[r1]->(b)-[r2]->(c)
                    WHERE b.name IS NOT NULL AND c.name IS NOT NULL AND c.name <> a.name
                    RETURN a.name AS src, b.name AS mid, c.name AS dst,
                           collect(DISTINCT type(r1)) AS rels1,
                           collect(DISTINCT type(r2)) AS rels2
                    ORDER BY mid ASC, dst ASC
                    """,
                    {"entity": entity},
                )
                for row in hop2_rows:
                    mid = row["mid"]
                    rel1 = _pick_preferred_rel(row.get("rels1", []))
                    rel2 = _pick_preferred_rel(row.get("rels2", []))
                    hop2_by_mid.setdefault(mid, []).append(
                        {
                            "hops": 2,
                            "chain_str": f"({row['src']}) --[{rel1}]--> ({mid}) --[{rel2}]--> ({row['dst']})",
                            "source": row["src"],
                            "rel1": rel1,
                            "intermediate": mid,
                            "rel2": rel2,
                            "target": row["dst"],
                        }
                    )

            chains: List[Dict[str, Any]] = []
            for row in hop1_rows:
                mid = row["mid"]
                rel1 = _pick_preferred_rel(row.get("rels1", []))
                chains.append(
                    {
                        "hops": 1,
                        "chain_str": f"({row['src']}) --[{rel1}]--> ({mid})",
                        "source": row["src"],
                        "rel1": rel1,
                        "intermediate": mid,
                        "rel2": None,
                        "target": mid,
                    }
                )
                if max_hops >= 2:
                    chains.extend(hop2_by_mid.get(mid, []))

            if chains or self.neo4j_error is None:
                return chains

        multi_hop_chains: List[Dict[str, Any]] = []
        if entity not in self.G:
            return multi_hop_chains

        for neighbor in self.G.successors(entity):
            rel1 = self.G[entity][neighbor].get("relation", "relates_to")
            multi_hop_chains.append(
                {
                    "hops": 1,
                    "chain_str": f"({entity}) --[{rel1}]--> ({neighbor})",
                    "source": entity,
                    "rel1": rel1,
                    "intermediate": neighbor,
                    "rel2": None,
                    "target": neighbor,
                }
            )

            if max_hops >= 2:
                for target_2 in self.G.successors(neighbor):
                    if target_2 != entity:
                        rel2 = self.G[neighbor][target_2].get("relation", "relates_to")
                        multi_hop_chains.append(
                            {
                                "hops": 2,
                                "chain_str": f"({entity}) --[{rel1}]--> ({neighbor}) --[{rel2}]--> ({target_2})",
                                "source": entity,
                                "rel1": rel1,
                                "intermediate": neighbor,
                                "rel2": rel2,
                                "target": target_2,
                            }
                        )

        return multi_hop_chains

    def _query_neo4j_safety_rules(self, egfr: float, hypertension: int) -> List[Dict[str, Any]]:
        """Truy xuất các chuỗi kiểm định dược lý trực tiếp từ đồ thị Neo4j."""
        rules: List[Dict[str, Any]] = []
        if not self.neo4j_connected:
            return rules

        if egfr < 30.0:
            rows = self.query_neo4j_cypher(
                """
                MATCH (m:Medication {name: "Metformin_First_Line"})-[r:CONTRAINDICATED_WITH|CHONG_CHI_DINH_KHI]->(b:Biomarker {name: "Severe_Renal_Impairment_eGFR_lt_30"})
                OPTIONAL MATCH (b)-[r2:REQUIRES|BAT_BUOC_CHUYEN_SANG]->(ins:Medication)
                RETURN m.name AS med, type(r) AS rel, b.name AS bio, r.hanh_dong AS action, ins.name AS alt_med
                LIMIT 1
                """
            )
            if rows:
                r0 = rows[0]
                rules.append(
                    {
                        "type": "CONTRAINDICATION",
                        "chain": f"({r0['med']}) --[{r0['rel']}]--> ({r0['bio']})",
                        "action": r0.get("action") or f"Ngừng Metformin, chuyển {r0.get('alt_med') or 'Insulin'}",
                    }
                )
        elif 30.0 <= egfr < 45.0:
            rows_red = self.query_neo4j_cypher(
                """
                MATCH (d:Disease {name: "Diabetes_Type_2"})-[r1:TREATED_BY|DIEU_TRI_DAU_TAY]->(m:Medication {name: "Metformin_First_Line"})-[r2:DOSE_REDUCTION_WHEN|GIAM_LIEU_KHI]->(b:Biomarker {name: "eGFR_30_to_45_Impairment"})
                RETURN d.name AS dis, type(r1) AS rel1, m.name AS med, type(r2) AS rel2, b.name AS bio, r2.hanh_dong AS action
                LIMIT 1
                """
            )
            if rows_red:
                r0 = rows_red[0]
                rules.append(
                    {
                        "type": "DOSE_REDUCTION",
                        "chain": f"({r0['dis']}) --[{r0['rel1']}]--> ({r0['med']}) --[{r0['rel2']}]--> ({r0['bio']})",
                        "action": r0.get("action") or "Giảm 50% liều Metformin",
                    }
                )

            rows_sglt = self.query_neo4j_cypher(
                """
                MATCH (d:Disease {name: "Diabetes_Type_2"})-[r1:TREATED_BY|DIEU_TRI_PHOI_HOP]->(m:Medication {name: "SGLT2_Inhibitor_Empagliflozin"})-[r2:PROVEN_BENEFIT|BAO_VE_CO_QUAN]->(t)
                RETURN d.name AS dis, type(r1) AS rel1, m.name AS med, type(r2) AS rel2, t.name AS target, r2.loi_ich AS benefit
                LIMIT 1
                """
            )
            if rows_sglt:
                r0 = rows_sglt[0]
                rules.append(
                    {
                        "type": "PROVEN_BENEFIT",
                        "chain": f"({r0['dis']}) --[{r0['rel1']}]--> ({r0['med']}) --[{r0['rel2']}]--> ({r0['target']})",
                        "action": r0.get("benefit") or "Ưu tiên phối hợp thuốc ức chế SGLT2 bảo vệ chức năng thận",
                    }
                )

        if hypertension == 1:
            rows_htn = self.query_neo4j_cypher(
                """
                MATCH (h:Disease {name: "Hypertension"})-[r1:TREATED_BY|DIEU_TRI_DAU_TAY]->(m:Medication {name: "ACE_Inhibitor"})
                RETURN h.name AS dis, type(r1) AS rel1, m.name AS med, r1.uu_diem AS action
                LIMIT 1
                """
            )
            if rows_htn:
                r0 = rows_htn[0]
                rules.append(
                    {
                        "type": "COMORBIDITY_CARDIO_RENAL",
                        "chain": f"({r0['dis']}) --[{r0['rel1']}]--> ({r0['med']})",
                        "action": r0.get("action") or "Phối hợp thuốc ức chế men chuyển để hạ áp và bảo vệ áp lực lọc cầu thận",
                    }
                )

        return rules

    def check_contraindications(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chốt chặn an toàn dược lý: Kiểm tra chống chỉ định thuốc và hiệu chỉnh liều
        bằng suy luận bắc cầu 2-hop dựa trên chức năng thận eGFR và bệnh nền.
        """
        egfr = float(patient.get("egfr", 80.0))
        hypertension = int(patient.get("hypertension", 0))

        detected_rules: List[Dict[str, Any]] = []
        contraindication_flag = False
        dose_adjustment = None
        recommended_drugs: List[str] = []
        avoid_drugs: List[str] = []

        if self.neo4j_connected:
            detected_rules = self._query_neo4j_safety_rules(egfr, hypertension)

        if egfr < 30.0:
            contraindication_flag = True
            avoid_drugs.append("Metformin_First_Line")
            dose_adjustment = (
                "CHỐNG CHỈ ĐỊNH TUYỆT ĐỐI Metformin (eGFR < 30 mL/phút). "
                "Ngừng thuốc ngay lập tức để phòng ngừa nhiễm toan Lactic Acidosis."
            )
            recommended_drugs.append("Insulin_Therapy")
            if not any(r["type"] == "CONTRAINDICATION" for r in detected_rules):
                detected_rules.append(
                    {
                        "type": "CONTRAINDICATION",
                        "chain": "(Metformin_First_Line) --[contraindicated_with]--> (Severe_Renal_Impairment_eGFR_lt_30)",
                        "action": "Ngừng Metformin, chuyển Insulin",
                    }
                )
        elif 30.0 <= egfr < 45.0:
            dose_adjustment = (
                f"GIẢM 50% LIỀU Metformin (eGFR = {egfr} mL/phút nằm trong khoảng 30 - 45 mL/phút). "
                "Liều tối đa 1000 mg/ngày, theo dõi eGFR định kỳ 3 tháng/lần."
            )
            recommended_drugs.append("Metformin (50% liều)")
            recommended_drugs.append("SGLT2_Inhibitor_Empagliflozin")
            if not any(r["type"] == "DOSE_REDUCTION" for r in detected_rules):
                detected_rules.append(
                    {
                        "type": "DOSE_REDUCTION",
                        "chain": "(Diabetes_Type_2) --[treated_by]--> (Metformin_First_Line) --[dose_reduction_when]--> (eGFR_30_to_45_mL_min)",
                        "action": "Giảm 50% liều Metformin",
                    }
                )
            if not any(r["type"] == "PROVEN_BENEFIT" for r in detected_rules):
                detected_rules.append(
                    {
                        "type": "PROVEN_BENEFIT",
                        "chain": "(Diabetes_Type_2) --[treated_by]--> (SGLT2_Inhibitor_Empagliflozin) --[proven_benefit]--> (Renal_and_Heart_Failure_Protection)",
                        "action": "Ưu tiên phối hợp thuốc ức chế SGLT2 bảo vệ chức năng thận",
                    }
                )
        else:
            dose_adjustment = "Sử dụng liều Metformin tiêu chuẩn (eGFR >= 45 mL/phút an toàn)."
            recommended_drugs.append("Metformin_First_Line")

        if hypertension == 1:
            recommended_drugs.append("ACE_Inhibitor")
            if not any(r["type"] == "COMORBIDITY_CARDIO_RENAL" for r in detected_rules):
                detected_rules.append(
                    {
                        "type": "COMORBIDITY_CARDIO_RENAL",
                        "chain": "(Hypertension) --[treated_by]--> (ACE_Inhibitor) --[proven_benefit]--> (Renal_Hemodynamic_Protection)",
                        "action": "Phối hợp thuốc ức chế men chuyển để hạ áp và bảo vệ áp lực lọc cầu thận",
                    }
                )

        return {
            "patient_id": patient.get("patient_id", "Unknown"),
            "egfr": egfr,
            "hypertension": hypertension,
            "contraindication_flag": contraindication_flag,
            "dose_adjustment": dose_adjustment,
            "recommended_drugs": recommended_drugs,
            "avoid_drugs": avoid_drugs,
            "detected_rules": detected_rules,
            "graph_backend": "Neo4j" if self.neo4j_connected else "NetworkX",
        }

    def find_matching_entities(self, query: str) -> List[str]:
        """Nhận diện các thực thể có trong đồ thị xuất hiện trong truy vấn."""
        query_lower = query.lower()

        if self.neo4j_connected:
            rows = self.query_neo4j_cypher(
                """
                MATCH (n)
                WHERE n.name IS NOT NULL
                RETURN n.name AS name, n.label AS label, n.ten_tieng_viet AS ten_tieng_viet
                """
            )
            if rows or self.neo4j_error is None:
                matched: List[str] = []
                for row in rows:
                    name = str(row.get("name", "")).strip()
                    if not name:
                        continue
                    label_raw = str(row.get("label") or "").lower()
                    vi_raw = str(row.get("ten_tieng_viet") or "").lower()
                    candidates = [
                        name.lower(),
                        name.replace("_", " ").lower(),
                        label_raw,
                        label_raw.split("(")[0].strip(),
                        vi_raw,
                        vi_raw.split("(")[0].strip(),
                    ]
                    if any(cand and len(cand) >= 3 and cand in query_lower for cand in candidates):
                        if name not in matched:
                            matched.append(name)
                return matched

        matched = []
        for node in self.G.nodes():
            if str(node).lower() in query_lower or str(node).replace("_", " ").lower() in query_lower:
                matched.append(str(node))
        return matched

    def get_graph_summary(self) -> Dict[str, Any]:
        """Thống kê kích thước đồ thị từ Neo4j (nếu đã kết nối) hoặc NetworkX fallback."""
        if self.neo4j_connected:
            node_rows = self.query_neo4j_cypher(
                "MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name ORDER BY n.name"
            )
            edge_rows = self.query_neo4j_cypher("MATCH ()-[r]->() RETURN count(r) AS edge_count")
            if self.neo4j_error is None and edge_rows:
                nodes = [str(r["name"]) for r in node_rows]
                return {
                    "num_nodes": len(nodes),
                    "num_edges": int(edge_rows[0]["edge_count"]),
                    "nodes": nodes,
                    "neo4j_connected": True,
                    "backend": "Neo4j",
                    "neo4j_uri": self.neo4j_uri,
                    "neo4j_error": None,
                }

        return {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "nodes": list(self.G.nodes()),
            "neo4j_connected": self.neo4j_connected,
            "backend": "NetworkX (Fallback)",
            "neo4j_uri": self.neo4j_uri,
            "neo4j_error": self.neo4j_error,
        }

    def generate_pyvis_html(self, output_html_path: str = "graph_visualization.html") -> str:
        """Tạo file HTML trực quan hóa đồ thị tương tác bằng PyVis từ Neo4j hoặc NetworkX."""
        try:
            from pyvis.network import Network

            net = Network(height="500px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)

            if self.neo4j_connected:
                node_rows = self.query_neo4j_cypher(
                    """
                    MATCH (n)
                    WHERE n.name IS NOT NULL
                    RETURN n.name AS id, labels(n) AS labels,
                           coalesce(n.label, n.ten_tieng_viet, n.name) AS display_label
                    """
                )
                edge_rows = self.query_neo4j_cypher(
                    """
                    MATCH (s)-[r]->(t)
                    WHERE s.name IS NOT NULL AND t.name IS NOT NULL
                    RETURN s.name AS src, t.name AS dst, collect(DISTINCT type(r)) AS rels
                    """
                )
                for row in node_rows:
                    n_id = str(row["id"])
                    labels = row.get("labels", [])
                    display_label = str(row.get("display_label") or n_id.replace("_", " "))
                    if "Disease" in labels or "BenhLy" in labels:
                        color, shape, size = "#FF4B4B", "dot", 26
                    elif "Biomarker" in labels or "ChiSoSinhHoc" in labels:
                        color, shape, size = "#00FFAA", "diamond", 20
                    elif "Medication" in labels or "ThuocDieuTri" in labels:
                        color, shape, size = "#2E86DE", "triangle", 22
                    elif "Complication" in labels or "BienChung" in labels:
                        color, shape, size = "#A855F7", "dot", 21
                    else:
                        color, shape, size = "#F39C12", "dot", 18
                    net.add_node(n_id, label=display_label, color=color, shape=shape, size=size, title=n_id)

                for row in edge_rows:
                    rel = _pick_preferred_rel(row.get("rels", []))
                    net.add_edge(str(row["src"]), str(row["dst"]), title=rel, label=rel, color="#8395A7", arrows="to")
            else:
                for node in self.G.nodes():
                    n_str = str(node)
                    if "Diabetes" in n_str or "Hypertension" in n_str:
                        color, shape, size = "#FF4B4B", "dot", 25
                    elif "Glucose" in n_str or "HbA1c" in n_str or "eGFR" in n_str:
                        color, shape, size = "#00FFAA", "diamond", 20
                    elif "Metformin" in n_str or "SGLT2" in n_str or "ACE" in n_str or "Insulin" in n_str:
                        color, shape, size = "#2E86DE", "triangle", 22
                    else:
                        color, shape, size = "#F39C12", "dot", 18

                    net.add_node(n_str, label=n_str.replace("_", " "), color=color, shape=shape, size=size)

                for u, v, data in self.G.edges(data=True):
                    rel = data.get("relation", "")
                    net.add_edge(str(u), str(v), title=rel, label=rel, color="#8395A7", arrows="to")

            net.set_options(
                """
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
            """
            )
            net.save_graph(output_html_path)
            return output_html_path
        except Exception as e:
            print(f"[!] Lỗi khi xuất đồ thị PyVis: {e}")
            return ""


if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    summary = kg.get_graph_summary()
    print(
        f"[+] Knowledge Graph ({summary['backend']}) sẵn sàng với "
        f"{summary['num_nodes']} nodes, {summary['num_edges']} edges."
    )

    sample_patient = {"patient_id": "BN001", "egfr": 42.0, "hypertension": 1}
    safety = kg.check_contraindications(sample_patient)
    print(f"[+] Kiểm tra an toàn BN001: {safety['dose_adjustment']}")
    for r in safety["detected_rules"]:
        print(f"    -> [{r['type']}]: {r['chain']}")
