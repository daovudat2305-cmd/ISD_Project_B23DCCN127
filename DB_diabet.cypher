// ============================================================================
// HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG - KHOA CÔNG NGHỆ THÔNG TIN 1
// HỌC PHẦN: PHÁT TRIỂN CÁC HỆ THỐNG THÔNG MINH (ISD 2026 - PGS.TS. TRẦN ĐÌNH QUẾ)
// DỰ ÁN 01: HỆ ĐA AGENT TƯ VẤN, CHẨN ĐOÁN VÀ QUẢN LÝ BỆNH TIỂU ĐƯỜNG
// TỆP CƠ SỞ TRI THỨC ĐỒ THỊ NEO4J CHUẨN TIẾNG VIỆT & DẪN CHỨNG: kb_diabet_neo4j.cypher
//
// NGUỒN DẪN CHỨNG CHÍNH THỐNG:
// 1. Bệnh viện ĐKQT Vinmec: https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/
// 2. Bộ Y tế Việt Nam: Quyết định số 5481/QĐ-BYT ban hành ngày 30/12/2020 (File: QD_5481_BYT.pdf)
// 3. Hiệp hội Đái tháo đường Hoa Kỳ: ADA Standards of Care in Diabetes 2024 (File: ADA_Standards_of_Care_2024.pdf)
// ============================================================================

// ----------------------------------------------------------------------------
// BƯỚC 1: DỌN SẠCH DỮ LIỆU CŨ TRONG DATABASE
// ----------------------------------------------------------------------------
MATCH (n) DETACH DELETE n;

// ----------------------------------------------------------------------------
// BƯỚC 2: TẠO RÀNG BUỘC DUY NHẤT (UNIQUE CONSTRAINT) ĐỂ TỐI ƯU TRUY VẤN
// ----------------------------------------------------------------------------
CREATE CONSTRAINT unique_concept_name IF NOT EXISTS
FOR (n:DiabetesConcept) REQUIRE n.name IS UNIQUE;

// ----------------------------------------------------------------------------
// BƯỚC 3: TẠO CÁC NÚT THỰC THỂ (NODES) CHUẨN TIẾNG VIỆT & METADATA
// Ghi chú: Sử dụng nhãn kép (bilingual labels) để tương thích cả tiếng Việt & tiếng Anh:
// :BenhLy / :Disease
// :TrieuChung / :Symptom
// :ChiSoSinhHoc / :Biomarker
// :ThuocDieuTri / :Medication
// :BienChung / :Complication
// :YeuToNguyCo / :RiskFactor
// :CanThiepLoiSong / :Intervention
// ----------------------------------------------------------------------------

// ==========================================
// 1. NHÓM BỆNH LÝ (DISEASE)
// ==========================================
CREATE (d1:Disease:BenhLy:DiabetesConcept {
    name: "Diabetes_Type_2",
    ten_tieng_viet: "Đái tháo đường Típ 2",
    label: "Đái tháo đường Típ 2",
    ma_icd10: "E11",
    dinh_nghia: "Bệnh rối loạn chuyển hóa đặc trưng bởi tăng glucose máu mạn tính do giảm tiết insulin, kháng insulin hoặc kết hợp cả hai.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec & Bộ Y tế (QĐ 5481/QĐ-BYT)",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Trang 5-8)",
    phan_loai: "Bệnh không lây nhiễm mạn tính"
});

CREATE (d2:Disease:BenhLy:DiabetesConcept {
    name: "Prediabetes",
    ten_tieng_viet: "Tiền đái tháo đường",
    label: "Tiền đái tháo đường",
    ma_icd10: "R73.0",
    dinh_nghia: "Tình trạng nồng độ glucose máu cao hơn bình thường nhưng chưa đạt ngưỡng chẩn đoán đái tháo đường (HbA1c 5.7 - 6.4%).",
    nguon_goc: "Bộ Y tế Việt Nam & ADA 2024",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 2.2)",
    phan_loai: "Giai đoạn cảnh báo sớm"
});

CREATE (d3:Disease:BenhLy:DiabetesConcept {
    name: "Hypertension",
    ten_tieng_viet: "Tăng huyết áp",
    label: "Tăng huyết áp",
    ma_icd10: "I10",
    dinh_nghia: "Huyết áp tâm thu >= 140 mmHg và/hoặc huyết áp tâm trương >= 90 mmHg, là bệnh lý đồng mắc thường gặp nhất ở bệnh nhân ĐTĐ.",
    nguon_goc: "Bệnh viện Vinmec & ADA 2024",
    link_web: "https://www.vinmec.com/vi/tim-mach/tang-huyet-ap/",
    tai_lieu_pdf: "ADA_Standards_of_Care_2024.pdf (Mục 10)",
    phan_loai: "Bệnh lý tim mạch đồng mắc"
});

// ==========================================
// 2. NHÓM TRIỆU CHỨNG LÂM SÀNG (SYMPTOM - VINMEC)
// ==========================================
CREATE (s1:Symptom:TrieuChung:DiabetesConcept {
    name: "Polyuria",
    ten_tieng_viet: "Tiểu nhiều (Đặc biệt ban đêm)",
    label: "Tiểu nhiều (Polyuria)",
    thuoc_hoi_chung: "Hội chứng 4 nhiều kinh điển",
    co_che: "Nồng độ glucose máu vượt ngưỡng tái hấp thu của ống thận (180 mg/dL), gây lợi niệu thẩm thấu kéo theo nước qua nước tiểu.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_TrieuChung_LamSang.pdf"
});

CREATE (s2:Symptom:TrieuChung:DiabetesConcept {
    name: "Polydipsia",
    ten_tieng_viet: "Khát nhiều & Uống nhiều",
    label: "Khát nhiều (Polydipsia)",
    thuoc_hoi_chung: "Hội chứng 4 nhiều kinh điển",
    co_che: "Mất nước nội bào và giảm thể tích tuần hoàn do tiểu nhiều kích thích liên tục trung tâm khát ở vùng dưới đồi.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_TrieuChung_LamSang.pdf"
});

CREATE (s3:Symptom:TrieuChung:DiabetesConcept {
    name: "Polyphagia",
    ten_tieng_viet: "Ăn nhiều mau đói",
    label: "Ăn nhiều (Polyphagia)",
    thuoc_hoi_chung: "Hội chứng 4 nhiều kinh điển",
    co_che: "Kháng insulin hoặc thiếu insulin khiến glucose không thể đi vào tế bào, tế bào phát tín hiệu bị đói năng lượng.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_TrieuChung_LamSang.pdf"
});

CREATE (s4:Symptom:TrieuChung:DiabetesConcept {
    name: "Weight_Loss",
    ten_tieng_viet: "Sụt cân nhanh không rõ nguyên nhân",
    label: "Sụt cân nhanh (Weight Loss)",
    thuoc_hoi_chung: "Hội chứng 4 nhiều kinh điển",
    co_che: "Do không dùng được glucose làm nhiên liệu, cơ thể buộc phải tăng phân giải mô mỡ và protein cơ bắp để cung cấp năng lượng.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_TrieuChung_LamSang.pdf"
});

CREATE (s5:Symptom:TrieuChung:DiabetesConcept {
    name: "Blurred_Vision",
    ten_tieng_viet: "Nhìn mờ mắt thoáng qua",
    label: "Nhìn mờ mắt",
    thuoc_hoi_chung: "Triệu chứng tăng áp lực thẩm thấu",
    co_che: "Tăng áp lực thẩm thấu làm thay đổi thể tích nước trong thể thủy tinh thể của mắt, gây biến dạng tiêu cự tạm thời.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_TrieuChung_LamSang.pdf"
});

CREATE (s6:Symptom:TrieuChung:DiabetesConcept {
    name: "Peripheral_Numbness",
    ten_tieng_viet: "Tê bì & Châm chích đầu ngón chi",
    label: "Tê bì đầu chi",
    thuoc_hoi_chung: "Triệu chứng thần kinh sớm",
    co_che: "Đường huyết cao kéo dài làm tổn thương các mạch máu nhỏ nuôi dây thần kinh ngoại biên.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_TrieuChung_LamSang.pdf"
});

CREATE (s7:Symptom:TrieuChung:DiabetesConcept {
    name: "Slow_Healing_Wounds",
    ten_tieng_viet: "Vết loét / Vết thương lâu lành",
    label: "Vết thương lâu lành",
    thuoc_hoi_chung: "Triệu chứng suy giảm miễn dịch & mạch máu",
    co_che: "Tổn thương vi tuần hoàn và suy giảm chức năng bạch cầu khiến mô thiếu máu nuôi và dễ nhiễm khuẩn dai dẳng.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_TrieuChung_LamSang.pdf"
});

// ==========================================
// 3. NHÓM CHỈ SỐ SINH HỌC & TIÊU CHUẨN CHẨN ĐOÁN (BIOMARKER - BYT 5481 & ADA)
// ==========================================
CREATE (b1:Biomarker:ChiSoSinhHoc:DiabetesConcept {
    name: "HbA1c_ge_6.5",
    ten_tieng_viet: "HbA1c >= 6.5%",
    label: "HbA1c >= 6.5%",
    nguong: 6.5,
    don_vi: "%",
    vai_tro: "Tiêu chuẩn vàng đánh giá kiểm soát đường huyết trung bình trong 3 tháng",
    nguon_goc: "Bộ Y tế Việt Nam (QĐ 5481/QĐ-BYT) & ADA 2024",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 2.1)"
});

CREATE (b2:Biomarker:ChiSoSinhHoc:DiabetesConcept {
    name: "Fasting_Glucose_ge_126",
    ten_tieng_viet: "Glucose máu lúc đói >= 126 mg/dL (7.0 mmol/L)",
    label: "FPG >= 126 mg/dL",
    nguong: 126.0,
    don_vi: "mg/dL",
    vai_tro: "Tiêu chuẩn chẩn đoán cơ bản sau nhịn ăn ít nhất 8 giờ",
    nguon_goc: "Bộ Y tế Việt Nam & Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 2.1)"
});

CREATE (b3:Biomarker:ChiSoSinhHoc:DiabetesConcept {
    name: "OGTT_2h_ge_200",
    ten_tieng_viet: "Nghiệm pháp dung nạp glucose đường uống 2 giờ >= 200 mg/dL",
    label: "OGTT 2h >= 200 mg/dL",
    nguong: 200.0,
    don_vi: "mg/dL",
    vai_tro: "Tiêu chuẩn chẩn đoán xác định sau uống dung dịch 75g glucose khan",
    nguon_goc: "Bộ Y tế Việt Nam & ADA 2024",
    link_web: "https://diabetesjournals.org/care/issue/47/Supplement_1",
    tai_lieu_pdf: "ADA_Standards_of_Care_2024.pdf (Table 2.2)"
});

CREATE (b4:Biomarker:ChiSoSinhHoc:DiabetesConcept {
    name: "Random_Glucose_ge_200",
    ten_tieng_viet: "Glucose máu ngẫu nhiên >= 200 mg/dL (11.1 mmol/L)",
    label: "Glucose ngẫu nhiên >= 200 mg/dL",
    nguong: 200.0,
    don_vi: "mg/dL",
    vai_tro: "Chẩn đoán xác định ngay lập tức nếu kèm triệu chứng 4 nhiều kinh điển",
    nguon_goc: "Bộ Y tế Việt Nam (QĐ 5481/QĐ-BYT)",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 2.1)"
});

CREATE (b5:Biomarker:ChiSoSinhHoc:DiabetesConcept {
    name: "eGFR_ge_60_Normal",
    ten_tieng_viet: "Độ lọc cầu thận eGFR >= 60 mL/phút/1.73m² (Chức năng thận bình thường)",
    label: "eGFR >= 60 (Thận bình thường)",
    nguong: 60.0,
    don_vi: "mL/phút/1.73m²",
    phan_tang: "CKD Giai đoạn 1-2",
    nguon_goc: "Bộ Y tế Việt Nam & ADA 2024",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 4.2.1)"
});

CREATE (b6:Biomarker:ChiSoSinhHoc:DiabetesConcept {
    name: "eGFR_30_to_45_Impairment",
    ten_tieng_viet: "Độ lọc cầu thận eGFR 30 - 44 mL/phút/1.73m² (Suy thận độ 3b)",
    label: "eGFR 30 - 44 (Suy thận độ 3b)",
    nguong_duoi: 30.0,
    nguong_tren: 44.0,
    don_vi: "mL/phút/1.73m²",
    phan_tang: "CKD Giai đoạn 3b (Suy thận mức độ vừa đến nặng)",
    nguon_goc: "Bộ Y tế Việt Nam (QĐ 5481) & ADA 2024",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 4.2.1)"
});

CREATE (b7:Biomarker:ChiSoSinhHoc:DiabetesConcept {
    name: "Severe_Renal_Impairment_eGFR_lt_30",
    ten_tieng_viet: "Độ lọc cầu thận eGFR < 30 mL/phút/1.73m² (Suy thận nặng độ 4-5)",
    label: "eGFR < 30 (Suy thận nặng)",
    nguong: 30.0,
    don_vi: "mL/phút/1.73m²",
    phan_tang: "CKD Giai đoạn 4-5 (Nguy cơ suy thận giai đoạn cuối)",
    nguon_goc: "Bộ Y tế Việt Nam (QĐ 5481) & ADA 2024",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 4.2.1)"
});

// ==========================================
// 4. NHÓM THUỐC ĐIỀU TRỊ (MEDICATION)
// ==========================================
CREATE (m1:Medication:ThuocDieuTri:DiabetesConcept {
    name: "Metformin_First_Line",
    ten_tieng_viet: "Metformin (Glucophage 500mg, 850mg, 1000mg)",
    label: "Metformin (Đầu tay)",
    nhom_duoc_ly: "Biguanide",
    co_che: "Ức chế tân tạo glucose ở gan và tăng độ nhạy cảm insulin ở cơ xương",
    lieu_toi_da_chuan: "2000 - 2500 mg/ngày",
    nguon_goc: "Bộ Y tế Việt Nam & ADA 2024 & Vinmec",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 4.1)"
});

CREATE (m2:Medication:ThuocDieuTri:DiabetesConcept {
    name: "SGLT2_Inhibitor_Empagliflozin",
    ten_tieng_viet: "Empagliflozin (Jardiance 10mg, 25mg)",
    label: "Empagliflozin (SGLT2i)",
    nhom_duoc_ly: "Ức chế kênh đồng vận chuyển Natri-Glucose 2 (SGLT2 Inhibitor)",
    co_che: "Tăng thải trừ glucose qua nước tiểu, giảm áp lực cầu thận và giảm gánh nặng tim mạch",
    uu_diem_vuot_troi: "Bảo vệ thận, làm chậm suy thận mạn và giảm tỷ lệ tử vong tim mạch",
    nguon_goc: "ADA Standards of Care 2024 & Vinmec",
    link_web: "https://diabetesjournals.org/care/issue/47/Supplement_1",
    tai_lieu_pdf: "ADA_Standards_of_Care_2024.pdf (Mục 9 & 11)"
});

CREATE (m3:Medication:ThuocDieuTri:DiabetesConcept {
    name: "Dapagliflozin",
    ten_tieng_viet: "Dapagliflozin (Forxiga 10mg)",
    label: "Dapagliflozin (SGLT2i)",
    nhom_duoc_ly: "Ức chế SGLT2",
    co_che: "Thải trừ đường qua đường niệu độc lập với insulin",
    nguon_goc: "Bộ Y tế Việt Nam & ADA 2024",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 4.1)"
});

CREATE (m4:Medication:ThuocDieuTri:DiabetesConcept {
    name: "Gliclazide_MR",
    ten_tieng_viet: "Gliclazide phóng thích kéo dài (Diamicron MR 30mg, 60mg)",
    label: "Gliclazide MR (Sulfonylurea)",
    nhom_duoc_ly: "Sulfonylurea thế hệ 2",
    co_che: "Kích thích tế bào beta đảo tụy tăng tiết insulin nội sinh",
    nguy_co: "Cần chú ý nguy cơ hạ đường huyết ở người già hoặc ăn uống kém",
    nguon_goc: "Bộ Y tế Việt Nam (QĐ 5481)",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 4.1)"
});

CREATE (m5:Medication:ThuocDieuTri:DiabetesConcept {
    name: "ACE_Inhibitor",
    ten_tieng_viet: "Thuốc ức chế men chuyển (Perindopril / Enalapril)",
    label: "Thuốc ức chế men chuyển (ACEi)",
    nhom_duoc_ly: "Ức chế hệ Renin-Angiotensin-Aldosterone (RAAS)",
    co_che: "Giãn tiểu động mạch đi cầu thận, giảm áp lực mao mạch cầu thận, kiểm soát huyết áp < 130/80 mmHg",
    nguon_goc: "Bệnh viện Vinmec & ADA 2024",
    link_web: "https://www.vinmec.com/vi/tim-mach/tang-huyet-ap/",
    tai_lieu_pdf: "ADA_Standards_of_Care_2024.pdf (Mục 10)"
});

CREATE (m6:Medication:ThuocDieuTri:DiabetesConcept {
    name: "Insulin_Basal",
    ten_tieng_viet: "Insulin nền (Insulin Glargine / Degludec)",
    label: "Insulin nền (Lantus/Tresiba)",
    nhom_duoc_ly: "Insulin tương tự tác dụng kéo dài",
    co_che: "Cung cấp lượng insulin nền sinh lý 24 giờ, bắt buộc khi tế bào beta tụy suy kiệt hoặc eGFR < 30",
    nguon_goc: "Bộ Y tế Việt Nam (QĐ 5481)",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 4.3)"
});

// ==========================================
// 5. NHÓM BIẾN CHỨNG (COMPLICATION)
// ==========================================
CREATE (c1:Complication:BienChung:DiabetesConcept {
    name: "Diabetic_Nephropathy",
    ten_tieng_viet: "Bệnh thận đái tháo đường & Suy thận mạn",
    label: "Biến chứng Thận (CKD)",
    co_che: "Tăng áp lực lọc và xơ hóa cầu thận dẫn đến albumin niệu và suy giảm tiến triển eGFR.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec & Bộ Y tế 5481",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 5.1)"
});

CREATE (c2:Complication:BienChung:DiabetesConcept {
    name: "Cardiovascular_Disease",
    ten_tieng_viet: "Biến cố tim mạch, Nhồi máu cơ tim & Suy tim",
    label: "Biến chứng Tim mạch (CVD)",
    co_che: "Xơ vữa động mạch tăng tốc và rối loạn chức năng nội mô do tăng đường huyết mạn tính.",
    nguon_goc: "ADA Standards of Care 2024 & Vinmec",
    link_web: "https://diabetesjournals.org/care/issue/47/Supplement_1",
    tai_lieu_pdf: "ADA_Standards_of_Care_2024.pdf (Mục 10)"
});

CREATE (c3:Complication:BienChung:DiabetesConcept {
    name: "Diabetic_Retinopathy",
    ten_tieng_viet: "Bệnh võng mạc đái tháo đường (Nguy cơ mù lòa)",
    label: "Biến chứng Mắt (Võng mạc)",
    co_che: "Vi phình mạch và xuất huyết vi mạch võng mạc do tổn thương mao mạch đáy mắt.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_BienChung_Mat.pdf"
});

CREATE (c4:Complication:BienChung:DiabetesConcept {
    name: "Diabetic_Foot_Ulcer",
    ten_tieng_viet: "Bàn chân đái tháo đường (Loét & Hoại tử chi)",
    label: "Biến chứng Bàn chân",
    co_che: "Kết hợp giữa bệnh thần kinh ngoại biên (mất cảm giác đau) và tắc nghẽn mạch máu ngoại vi.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_ChamSoc_BanChan.pdf"
});

CREATE (c5:Complication:BienChung:DiabetesConcept {
    name: "Lactic_Acidosis",
    ten_tieng_viet: "Nhiễm toan Lactic (Cực kỳ nguy hiểm, tử vong cao)",
    label: "Nhiễm toan Lactic",
    co_che: "Tích lũy nồng độ Metformin trong máu khi suy thận nặng (eGFR < 30) ức chế chuyển hóa lactate ở gan.",
    nguon_goc: "Bộ Y tế Việt Nam & FDA Cảnh báo",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 4.2.1)"
});

// ==========================================
// 6. NHÓM YẾU TỐ NGUY CƠ (RISK FACTOR)
// ==========================================
CREATE (r1:RiskFactor:YeuToNguyCo:DiabetesConcept {
    name: "Obesity_BMI_ge_23",
    ten_tieng_viet: "Thừa cân / Béo phì (BMI >= 23 kg/m² theo chuẩn Châu Á)",
    label: "Béo phì (BMI >= 23)",
    nguong: 23.0,
    don_vi: "kg/m²",
    nguon_goc: "Bệnh viện ĐKQT Vinmec & WHO Tây Thái Bình Dương",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Tieu_Chuan_BMI_Chau_A.pdf"
});

CREATE (r2:RiskFactor:YeuToNguyCo:DiabetesConcept {
    name: "Sedentary_Lifestyle",
    ten_tieng_viet: "Lối sống tĩnh tại, ít vận động thể lực",
    label: "Ít vận động thể lực",
    nguon_goc: "Bệnh viện ĐKQT Vinmec & ADA 2024",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "ADA_Standards_of_Care_2024.pdf (Mục 5)"
});

CREATE (r3:RiskFactor:YeuToNguyCo:DiabetesConcept {
    name: "Family_History_Diabetes",
    ten_tieng_viet: "Tiền sử gia đình thế hệ thứ nhất (cha mẹ, anh chị em) mắc ĐTĐ",
    label: "Tiền sử gia đình mắc ĐTĐ",
    nguon_goc: "Bệnh viện ĐKQT Vinmec & Bộ Y tế 5481",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 1.2)"
});

// ==========================================
// 7. NHÓM CAN THIỆP LỐI SỐNG & PHÒNG NGỪA (INTERVENTION)
// ==========================================
CREATE (i1:Intervention:CanThiepLoiSong:DiabetesConcept {
    name: "Low_GI_Diet_Plan",
    ten_tieng_viet: "Dinh dưỡng: Cắt giảm 35% tinh bột nhanh (High-GI), tăng rau xanh giàu chất xơ",
    label: "Ăn giảm 35% Carb High-GI",
    loi_khuyen_chat: "Hạn chế cơm trắng, bánh mì trắng, nước ngọt có ga; ưu tiên gạo lứt, ngũ cốc nguyên cám và ăn rau trước khi ăn tinh bột.",
    nguon_goc: "Khoa Dinh dưỡng - Bệnh viện ĐKQT Vinmec",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "Vinmec_CheDoAn_TieuDuong.pdf"
});

CREATE (i2:Intervention:CanThiepLoiSong:DiabetesConcept {
    name: "Brisk_Walking_30min",
    ten_tieng_viet: "Vận động: Đi bộ nhanh 30 phút sau bữa ăn chính",
    label: "Đi bộ nhanh 30 phút sau ăn",
    loi_khuyen_chat: "Tập luyện cách bữa ăn 15-30 phút kích hoạt thụ thể vận chuyển GLUT4 đưa glucose vào tế bào độc lập với insulin, cắt cơn tăng đường huyết sau ăn.",
    nguon_goc: "Bệnh viện ĐKQT Vinmec & ADA 2024",
    link_web: "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
    tai_lieu_pdf: "ADA_Standards_of_Care_2024.pdf (Mục 5)"
});

CREATE (i3:Intervention:CanThiepLoiSong:DiabetesConcept {
    name: "Renal_Cardio_Protection",
    ten_tieng_viet: "Bảo vệ thận và tim mạch sớm bằng thuốc ức chế SGLT2 & ACEi",
    label: "Bảo tồn chức năng lọc thận",
    loi_khuyen_chat: "Giảm áp lực lọc nội cầu thận, làm chậm tiến trình chuyển sang suy thận giai đoạn cuối phải chạy thận nhân tạo.",
    nguon_goc: "ADA Standards of Care 2024 & Bộ Y tế 5481",
    link_web: "https://diabetesjournals.org/care/issue/47/Supplement_1",
    tai_lieu_pdf: "ADA_Standards_of_Care_2024.pdf (Mục 11)"
});

CREATE (i4:Intervention:CanThiepLoiSong:DiabetesConcept {
    name: "Periodic_Monitoring_3Months",
    ten_tieng_viet: "Tái khám định kỳ mỗi 3 tháng xét nghiệm HbA1c và eGFR",
    label: "Tái khám định kỳ 3 tháng/lần",
    loi_khuyen_chat: "Theo dõi sát chức năng thận eGFR để kịp thời điều chỉnh liều thuốc, ngăn ngừa biến chứng nhiễm độc thuốc.",
    nguon_goc: "Bộ Y tế Việt Nam (QĐ 5481/QĐ-BYT)",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Mục 6.2)"
});

// ----------------------------------------------------------------------------
// BƯỚC 4: TẠO CÁC CẠNH QUAN HỆ NGỮ NGHĨA (RELATIONSHIPS)
// Bao gồm cả nhãn quan hệ tiếng Việt & tiếng Anh để tương thích 100%
// ----------------------------------------------------------------------------

// A. Quan hệ Chẩn đoán (Disease -> Biomarker)
MATCH (d:Disease {name: "Diabetes_Type_2"}), (b:Biomarker {name: "HbA1c_ge_6.5"})
CREATE (d)-[:CHAN_DOAN_BOI {ten_quan_he: "Chẩn đoán bởi", tieu_chuan: "HbA1c >= 6.5%", nguon: "Bộ Y tế 5481 & ADA 2024"}]->(b),
       (d)-[:DIAGNOSED_BY]->(b);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (b:Biomarker {name: "Fasting_Glucose_ge_126"})
CREATE (d)-[:CHAN_DOAN_BOI {ten_quan_he: "Chẩn đoán bởi", tieu_chuan: "Đường huyết đói >= 126 mg/dL", nguon: "Vinmec & Bộ Y tế 5481"}]->(b),
       (d)-[:DIAGNOSED_BY]->(b);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (b:Biomarker {name: "OGTT_2h_ge_200"})
CREATE (d)-[:CHAN_DOAN_BOI {ten_quan_he: "Chẩn đoán bởi", tieu_chuan: "Dung nạp glucose 2h >= 200 mg/dL", nguon: "ADA 2024"}]->(b),
       (d)-[:DIAGNOSED_BY]->(b);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (b:Biomarker {name: "Random_Glucose_ge_200"})
CREATE (d)-[:CHAN_DOAN_BOI {ten_quan_he: "Chẩn đoán bởi", tieu_chuan: "Đường huyết ngẫu nhiên >= 200 mg/dL kèm triệu chứng", nguon: "Bộ Y tế 5481"}]->(b),
       (d)-[:DIAGNOSED_BY]->(b);

// B. Quan hệ Triệu chứng Lâm sàng (Disease -> Symptom - Hội chứng 4 nhiều Vinmec)
MATCH (d:Disease {name: "Diabetes_Type_2"}), (s:Symptom {name: "Polyuria"})
CREATE (d)-[:CO_TRIEU_CHUNG {ten_quan_he: "Có triệu chứng", thuoc_hoi_chung: "Hội chứng 4 nhiều", nguon: "Vinmec"}]->(s),
       (d)-[:HAS_SYMPTOM]->(s);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (s:Symptom {name: "Polydipsia"})
CREATE (d)-[:CO_TRIEU_CHUNG {ten_quan_he: "Có triệu chứng", thuoc_hoi_chung: "Hội chứng 4 nhiều", nguon: "Vinmec"}]->(s),
       (d)-[:HAS_SYMPTOM]->(s);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (s:Symptom {name: "Polyphagia"})
CREATE (d)-[:CO_TRIEU_CHUNG {ten_quan_he: "Có triệu chứng", thuoc_hoi_chung: "Hội chứng 4 nhiều", nguon: "Vinmec"}]->(s),
       (d)-[:HAS_SYMPTOM]->(s);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (s:Symptom {name: "Weight_Loss"})
CREATE (d)-[:CO_TRIEU_CHUNG {ten_quan_he: "Có triệu chứng", thuoc_hoi_chung: "Hội chứng 4 nhiều", nguon: "Vinmec"}]->(s),
       (d)-[:HAS_SYMPTOM]->(s);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (s:Symptom {name: "Blurred_Vision"})
CREATE (d)-[:CO_TRIEU_CHUNG {ten_quan_he: "Có triệu chứng", mo_ta: "Nhìn mờ thể thủy tinh", nguon: "Vinmec"}]->(s),
       (d)-[:HAS_SYMPTOM]->(s);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (s:Symptom {name: "Peripheral_Numbness"})
CREATE (d)-[:CO_TRIEU_CHUNG {ten_quan_he: "Có triệu chứng", mo_ta: "Tê bì châm chích ngón chân", nguon: "Vinmec"}]->(s),
       (d)-[:HAS_SYMPTOM]->(s);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (s:Symptom {name: "Slow_Healing_Wounds"})
CREATE (d)-[:CO_TRIEU_CHUNG {ten_quan_he: "Có triệu chứng", mo_ta: "Vết thương khó lành", nguon: "Vinmec"}]->(s),
       (d)-[:HAS_SYMPTOM]->(s);

// C. Quan hệ Yếu tố Nguy cơ (Disease -> RiskFactor)
MATCH (d:Disease {name: "Diabetes_Type_2"}), (r:RiskFactor {name: "Obesity_BMI_ge_23"})
CREATE (d)-[:CO_YEU_TO_NGUY_CO {ten_quan_he: "Có yếu tố nguy cơ", chi_tieu: "BMI >= 23 kg/m²", nguon: "Vinmec"}]->(r),
       (d)-[:HAS_RISK_FACTOR]->(r);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (r:RiskFactor {name: "Sedentary_Lifestyle"})
CREATE (d)-[:CO_YEU_TO_NGUY_CO {ten_quan_he: "Có yếu tố nguy cơ", chi_tieu: "Ít vận động thể lực", nguon: "Vinmec & ADA"}]->(r),
       (d)-[:HAS_RISK_FACTOR]->(r);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (r:RiskFactor {name: "Family_History_Diabetes"})
CREATE (d)-[:CO_YEU_TO_NGUY_CO {ten_quan_he: "Có yếu tố nguy cơ", chi_tieu: "Di truyền gia đình", nguon: "Bộ Y tế 5481"}]->(r),
       (d)-[:HAS_RISK_FACTOR]->(r);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (d3:Disease {name: "Hypertension"})
CREATE (d)-[:DONG_MAC_VOI {ten_quan_he: "Bệnh lý đồng mắc", nguon: "Vinmec & ADA 2024"}]->(d3),
       (d)-[:COMORBID_WITH]->(d3);

// D. Quan hệ Biến chứng (Disease -> Complication)
MATCH (d:Disease {name: "Diabetes_Type_2"}), (c:Complication {name: "Diabetic_Nephropathy"})
CREATE (d)-[:GAY_BIEN_CHUNG {ten_quan_he: "Gây biến chứng", co_quan: "Thận", nguon: "Vinmec & Bộ Y tế 5481"}]->(c),
       (d)-[:CAUSES_COMPLICATION]->(c);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (c:Complication {name: "Cardiovascular_Disease"})
CREATE (d)-[:GAY_BIEN_CHUNG {ten_quan_he: "Gây biến chứng", co_quan: "Tim mạch", nguon: "Vinmec & ADA 2024"}]->(c),
       (d)-[:CAUSES_COMPLICATION]->(c);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (c:Complication {name: "Diabetic_Retinopathy"})
CREATE (d)-[:GAY_BIEN_CHUNG {ten_quan_he: "Gây biến chứng", co_quan: "Mắt", nguon: "Vinmec"}]->(c),
       (d)-[:CAUSES_COMPLICATION]->(c);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (c:Complication {name: "Diabetic_Foot_Ulcer"})
CREATE (d)-[:GAY_BIEN_CHUNG {ten_quan_he: "Gây biến chứng", co_quan: "Bàn chân", nguon: "Vinmec"}]->(c),
       (d)-[:CAUSES_COMPLICATION]->(c);

// E. Quan hệ Phác đồ Điều trị Thuốc (Disease -> Medication)
MATCH (d:Disease {name: "Diabetes_Type_2"}), (m:Medication {name: "Metformin_First_Line"})
CREATE (d)-[:DIEU_TRI_DAU_TAY {ten_quan_he: "Điều trị đầu tay", uu_tien: 1, huong_dan: "Thuốc lựa chọn đầu tay khi không có chống chỉ định suy thận nặng", nguon: "Bộ Y tế 5481 & ADA 2024"}]->(m),
       (d)-[:TREATED_BY]->(m);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (m:Medication {name: "SGLT2_Inhibitor_Empagliflozin"})
CREATE (d)-[:DIEU_TRI_PHOI_HOP {ten_quan_he: "Điều trị phối hợp ưu tiên", uu_tien: 2, chi_dinh: "Ưu tiên khi có bệnh thận mạn CKD hoặc nguy cơ tim mạch cao", nguon: "ADA 2024"}]->(m),
       (d)-[:TREATED_BY]->(m);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (m:Medication {name: "Dapagliflozin"})
CREATE (d)-[:DIEU_TRI_PHOI_HOP {ten_quan_he: "Điều trị phối hợp", uu_tien: 2, nguon: "Bộ Y tế 5481"}]->(m),
       (d)-[:TREATED_BY]->(m);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (m:Medication {name: "Gliclazide_MR"})
CREATE (d)-[:DIEU_TRI_PHOI_HOP {ten_quan_he: "Điều trị phối hợp", uu_tien: 3, chi_dinh: "Phối hợp khi Metformin đơn trị liệu không đạt mục tiêu HbA1c", nguon: "Bộ Y tế 5481"}]->(m),
       (d)-[:TREATED_BY]->(m);

// F. CẠNH QUAN HỆ CỐT TỬ 2-HOP: DƯỢC LÝ & AN TOÀN THẬN (Medication -> Biomarker)
// Mắt xích 1: Giảm 50% liều Metformin khi eGFR từ 30 đến 44 mL/phút
MATCH (m:Medication {name: "Metformin_First_Line"}), (b:Biomarker {name: "eGFR_30_to_45_Impairment"})
CREATE (m)-[:GIAM_LIEU_KHI {
    ten_quan_he: "Giảm liều khi",
    hanh_dong: "BẮT BUỘC GIẢM 50% LIỀU (Không quá 1000 mg/ngày)",
    theo_doi: "Kiểm tra chức năng thận eGFR định kỳ 3 tháng/lần",
    nguon: "Quyết định 5481/QĐ-BYT Mục 4.2.1 & ADA 2024",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Trang 21)"
}]->(b),
(m)-[:DOSE_REDUCTION_WHEN]->(b);

// Mắt xích 2: Chống chỉ định tuyệt đối Metformin khi eGFR < 30 mL/phút
MATCH (m:Medication {name: "Metformin_First_Line"}), (b:Biomarker {name: "Severe_Renal_Impairment_eGFR_lt_30"})
CREATE (m)-[:CHONG_CHI_DINH_KHI {
    ten_quan_he: "Chống chỉ định tuyệt đối khi",
    hanh_dong: "NGỪNG METFORMIN NGAY LẬP TỨC! Chuyển sang phác đồ Insulin",
    canh_bao_tu_vong: "Nguy cơ tích lũy thuốc gây nhiễm toan Lactic Acidosis tử vong",
    nguon: "Quyết định 5481/QĐ-BYT Mục 4.2.1 & FDA Warning",
    link_web: "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
    tai_lieu_pdf: "QD_5481_BYT.pdf (Trang 21)"
}]->(b),
(m)-[:CONTRAINDICATED_WITH]->(b);

// Nối nguy cơ biến chứng nhiễm toan Lactic từ suy thận nặng
MATCH (b:Biomarker {name: "Severe_Renal_Impairment_eGFR_lt_30"}), (c:Complication {name: "Lactic_Acidosis"})
CREATE (b)-[:NGUY_CO_DANG_KE {ten_quan_he: "Nguy cơ đáng kể gây", mo_ta: "Nhiễm toan Lactic khi dùng Metformin", nguon: "Bộ Y tế 5481"}]->(c);

// Bắt buộc chuyển sang Insulin khi eGFR < 30
MATCH (b:Biomarker {name: "Severe_Renal_Impairment_eGFR_lt_30"}), (m:Medication {name: "Insulin_Basal"})
CREATE (b)-[:BAT_BUOC_CHUYEN_SANG {ten_quan_he: "Bắt buộc chuyển sang", chi_dinh: "Phác đồ Insulin kiểm soát đường huyết khi suy thận nặng", nguon: "Bộ Y tế 5481"}]->(m),
       (b)-[:REQUIRES]->(m);

// G. Bảo vệ Tim Thận & Huyết áp (Medication -> Intervention / Disease)
MATCH (m:Medication {name: "SGLT2_Inhibitor_Empagliflozin"}), (i:Intervention {name: "Renal_Cardio_Protection"})
CREATE (m)-[:BAO_VE_CO_QUAN {
    ten_quan_he: "Bảo vệ cơ quan đích",
    loi_ich: "Bảo tồn chức năng lọc thận và giảm biến cố suy tim ứ huyết",
    bang_chung: "Thử nghiệm lâm sàng EMPA-REG OUTCOME & EMPA-KIDNEY",
    nguon: "ADA Standards of Care 2024",
    link_web: "https://diabetesjournals.org/care/issue/47/Supplement_1"
}]->(i),
(m)-[:PROVEN_BENEFIT]->(i);

MATCH (d3:Disease {name: "Hypertension"}), (m:Medication {name: "ACE_Inhibitor"})
CREATE (d3)-[:DIEU_TRI_DAU_TAY {ten_quan_he: "Điều trị ưu tiên", uu_diem: "Hạ huyết áp mục tiêu < 130/80 mmHg và bảo vệ áp lực nội mao mạch cầu thận", nguon: "Vinmec & ADA 2024"}]->(m),
       (d3)-[:TREATED_BY]->(m);

// H. Quan hệ Can thiệp Lối sống & Dinh dưỡng (Disease -> Intervention)
MATCH (d:Disease {name: "Diabetes_Type_2"}), (i:Intervention {name: "Low_GI_Diet_Plan"})
CREATE (d)-[:CAN_THIEP_LOI_SONG {ten_quan_he: "Can thiệp dinh dưỡng", chi_tiet: "Cắt giảm 35% tinh bột nhanh, ăn nhiều chất xơ", nguon: "Vinmec"}]->(i),
       (d)-[:MANAGED_BY]->(i);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (i:Intervention {name: "Brisk_Walking_30min"})
CREATE (d)-[:CAN_THIEP_LOI_SONG {ten_quan_he: "Can thiệp vận động", chi_tiet: "Đi bộ nhanh 30 phút sau bữa ăn chính", nguon: "Vinmec & ADA"}]->(i),
       (d)-[:MANAGED_BY]->(i);

MATCH (d:Disease {name: "Diabetes_Type_2"}), (i:Intervention {name: "Periodic_Monitoring_3Months"})
CREATE (d)-[:THEO_DOI_DINH_KY {ten_quan_he: "Theo dõi định kỳ", chi_tiet: "Xét nghiệm HbA1c và eGFR 3 tháng/lần", nguon: "Bộ Y tế 5481"}]->(i),
       (d)-[:MANAGED_BY]->(i);

// ============================================================================
// BƯỚC 5: CÁC CÂU TRUY VẤN MẪU ĐỂ SHOW DẠNG GRAPH TRÊN NEO4J BROWSER
// ============================================================================

// 1. TRUY VẤN XEM TOÀN BỘ ĐỒ THỊ DẠNG MẠNG LƯỚI GRAPH (SHOW CHO GIẢNG VIÊN):
// MATCH (n:DiabetesConcept)-[r]->(m:DiabetesConcept)
// RETURN n, r, m LIMIT 100;

// 2. TRUY VẤN XEM HỘI CHỨNG 4 NHIỀU CỦA VINMEC (TRIỆU CHỨNG LÂM SÀNG):
// MATCH path = (d:BenhLy {name: "Diabetes_Type_2"})-[:CO_TRIEU_CHUNG]->(s:TrieuChung)
// RETURN path;

// 3. TRUY VẤN SUY LUẬN 2-HOP: PHÁC ĐỒ DƯỢC LÝ & AN TOÀN THẬN CỦA METFORMIN (CỐT TỬ CỦA ĐỒ ÁN):
// MATCH path = (d:BenhLy {name: "Diabetes_Type_2"})-[:DIEU_TRI_DAU_TAY]->(m:ThuocDieuTri)-[:GIAM_LIEU_KHI|CHONG_CHI_DINH_KHI]->(b:ChiSoSinhHoc)
// RETURN path;

// 4. TRUY VẤN PHÁC ĐỒ PHỐI HỢP THUỐC BẢO VỆ TIM THẬN (SGLT2i + ACEi):
// MATCH path = (m:ThuocDieuTri)-[:BAO_VE_CO_QUAN]->(i:CanThiepLoiSong)
// RETURN path;

// 5. TRUY VẤN BẢNG DẪN CHỨNG LINK WEB VINMEC VÀ TỆP PDF BỘ Y TẾ 5481:
// MATCH (n:DiabetesConcept)
// WHERE n.link_web IS NOT NULL
// RETURN n.ten_tieng_viet AS KhaiNiemYTe, n.nguon_goc AS NguonXacThuc, n.link_web AS LinkWebsite, n.tai_lieu_pdf AS FilePDFThamChieu;
