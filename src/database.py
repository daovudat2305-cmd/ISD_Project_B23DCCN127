"""
Module: src/database.py
Mục đích: Khởi tạo và quản trị Cơ sở dữ liệu SQLite (medical_intelligent_system.db)
Phân hệ: Khối 1 - Nguồn dữ liệu (EHR, CGM Telemetry Stream, Audit Action Logs)
Học phần: Phát triển các hệ thống thông minh - Học viện CN Bưu chính Viễn thông
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medical_intelligent_system.db")

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_database(db_path: str = DB_PATH) -> None:
    """Khởi tạo cấu trúc 3 bảng CSDL và nạp sẵn 3 hồ sơ bệnh nhân chuẩn mực."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Bảng 1: Hồ sơ bệnh án điện tử (EHR)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        bmi REAL NOT NULL,
        hba1c REAL NOT NULL,
        fasting_glucose REAL NOT NULL,
        systolic_bp INTEGER NOT NULL,
        diastolic_bp INTEGER NOT NULL,
        egfr REAL NOT NULL,
        hypertension INTEGER NOT NULL,
        heart_disease INTEGER NOT NULL,
        smoking_history TEXT NOT NULL,
        clinical_notes TEXT
    )
    """)

    # Bảng 2: Luồng dữ liệu đường huyết liên tục 12h (CGM Telemetry Stream)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cgm_telemetry_stream (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        hour_offset INTEGER NOT NULL,
        glucose_mgdl REAL NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
    """)

    # Bảng 3: Nhật ký kiểm toán tác tử (Agent Action Logs)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_action_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        phase TEXT NOT NULL,
        agent_name TEXT NOT NULL,
        ibm_level TEXT NOT NULL,
        action_taken TEXT NOT NULL,
        observation_detail TEXT,
        groundedness_score REAL,
        status TEXT NOT NULL
    )
    """)

    # Kiểm tra nếu chưa có dữ liệu bệnh nhân thì khởi tạo
    cursor.execute("SELECT COUNT(*) FROM patients")
    if cursor.fetchone()[0] == 0:
        sample_patients = [
            (
                "BN001", "Nguyễn Văn An", 58, "Nam", 31.2, 7.8, 192.0, 145, 90, 42.0, 1, 0,
                "former", "Tiểu đường típ 2, Tăng huyết áp độ 1, Suy thận mạn độ 3b (eGFR=42.0 mL/phút)"
            ),
            (
                "BN002", "Trần Thị Bình", 46, "Nữ", 26.5, 6.1, 115.0, 125, 80, 88.0, 0, 0,
                "never", "Tiền tiểu đường, chức năng thận bình thường, thể trạng thừa cân nhẹ"
            ),
            (
                "BN003", "Lê Hoàng Cường", 67, "Nam", 29.8, 8.6, 210.0, 155, 95, 28.0, 1, 1,
                "current", "Tiểu đường kiểm soát kém, Tăng huyết áp độ 2, Suy thận nặng độ 4 (eGFR=28.0 mL/phút), nguy cơ nhiễm toan lactic cao"
            )
        ]
        cursor.executemany("""
        INSERT INTO patients (
            patient_id, full_name, age, gender, bmi, hba1c, fasting_glucose,
            systolic_bp, diastolic_bp, egfr, hypertension, heart_disease,
            smoking_history, clinical_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_patients)

        # Khởi tạo chuỗi CGM 12 giờ trước (mỗi giờ 1 điểm đo)
        cgm_records = []
        now = datetime.now()
        
        # BN001: Đường huyết tăng dần từ 145 lên 196 mg/dL
        bn001_cgm = [145.0, 150.0, 158.0, 165.0, 172.0, 180.0, 177.0, 183.0, 188.0, 186.0, 192.0, 196.0]
        for h, val in enumerate(bn001_cgm):
            t_str = (now - timedelta(hours=12 - h)).strftime("%Y-%m-%d %H:%M:%S")
            cgm_records.append(("BN001", t_str, h - 12, val))

        # BN002: Dao động ổn định trong vùng 105 - 120 mg/dL
        bn002_cgm = [105.0, 108.0, 112.0, 110.0, 118.0, 115.0, 109.0, 114.0, 116.0, 111.0, 113.0, 115.0]
        for h, val in enumerate(bn002_cgm):
            t_str = (now - timedelta(hours=12 - h)).strftime("%Y-%m-%d %H:%M:%S")
            cgm_records.append(("BN002", t_str, h - 12, val))

        # BN003: Đường huyết rất cao dao động 195 - 225 mg/dL
        bn003_cgm = [195.0, 202.0, 208.0, 215.0, 210.0, 222.0, 218.0, 225.0, 219.0, 214.0, 210.0, 218.0]
        for h, val in enumerate(bn003_cgm):
            t_str = (now - timedelta(hours=12 - h)).strftime("%Y-%m-%d %H:%M:%S")
            cgm_records.append(("BN003", t_str, h - 12, val))

        cursor.executemany("""
        INSERT INTO cgm_telemetry_stream (patient_id, timestamp, hour_offset, glucose_mgdl)
        VALUES (?, ?, ?, ?)
        """, cgm_records)

    conn.commit()
    conn.close()

def get_patient(patient_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_all_patients(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY patient_id")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_cgm_stream(patient_id: str, db_path: str = DB_PATH) -> List[float]:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT glucose_mgdl FROM cgm_telemetry_stream
    WHERE patient_id = ?
    ORDER BY hour_offset ASC
    """, (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

def log_agent_action(
    patient_id: str,
    phase: str,
    agent_name: str,
    ibm_level: str,
    action_taken: str,
    observation_detail: str,
    groundedness_score: float = 1.0,
    status: str = "SUCCESS",
    db_path: str = DB_PATH
) -> None:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO agent_action_logs (
        timestamp, patient_id, phase, agent_name, ibm_level,
        action_taken, observation_detail, groundedness_score, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        patient_id, phase, agent_name, ibm_level,
        action_taken, observation_detail, groundedness_score, status
    ))
    conn.commit()
    conn.close()

def get_agent_logs(patient_id: Optional[str] = None, limit: int = 50, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    if patient_id:
        cursor.execute("""
        SELECT * FROM agent_action_logs
        WHERE patient_id = ?
        ORDER BY id DESC LIMIT ?
        """, (patient_id, limit))
    else:
        cursor.execute("""
        SELECT * FROM agent_action_logs
        ORDER BY id DESC LIMIT ?
        """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    init_database()
    print("[+] Khởi tạo SQLite medical_intelligent_system.db thành công!")
    p1 = get_patient("BN001")
    print(f"[*] Bệnh nhân BN001: {p1['full_name']} | Tuổi: {p1['age']} | BMI: {p1['bmi']} | eGFR: {p1['egfr']}")
    cgm = get_cgm_stream("BN001")
    print(f"[*] Luồng CGM 12h: {cgm}")

