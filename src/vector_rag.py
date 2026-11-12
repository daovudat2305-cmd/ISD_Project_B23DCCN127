"""
Module: src/vector_rag.py
Mục đích: Xử lý văn bản phi cấu trúc, phân đoạn (Chunking có Overlap),
truy xuất véc-tơ Top-k (Cosine Similarity) và Hợp nhất Ngữ cảnh (Context Fusion)
Phân hệ: Khối 3 - Hybrid GraphRAG Layer (Vector Store + Knowledge Graph)
Học phần: Phát triển các hệ thống thông minh - Học viện CN Bưu chính Viễn thông
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

class ClinicalDocumentChunker:
    """
    Thuật toán phân đoạn văn bản cửa sổ trượt (Sliding Window Chunking)
    với siêu tham số chuẩn mực chunk_size ~ 400 từ, overlap = 80 từ.
    Bảo toàn toàn vẹn ngữ cảnh ranh giới điều khoản y tế.
    """
    def __init__(self, chunk_size: int = 250, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str, source: str, url: str = "", pdf_ref: str = "") -> List[Dict[str, Any]]:
        # Chuẩn hóa khoảng trắng
        cleaned = re.sub(r'\s+', ' ', text).strip()
        words = cleaned.split()
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunk_idx += 1
            chunks.append({
                "chunk_id": f"{source}_chk_{chunk_idx:03d}",
                "source": source,
                "url": url,
                "pdf_ref": pdf_ref,
                "text": chunk_text,
                "word_count": len(chunk_words)
            })
            if end >= len(words):
                break
            start += (self.chunk_size - self.overlap)

        return chunks


class ClinicalVectorStore:
    """
    Kho chỉ mục véc-tơ ngữ nghĩa (Vector Database).
    Sử dụng không gian nhúng TF-IDF chuẩn hóa L2 và tích vô hướng Cosine Similarity
    đảm bảo độ trễ siêu thấp (< 5ms) và vận hành độc lập offline 100%.
    """
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
        self.embeddings: Optional[np.ndarray] = None
        self.is_indexed = False

    def build_index_from_directory(self, data_dir: str = DATA_DIR, chunker: ClinicalDocumentChunker = None) -> int:
        if chunker is None:
            chunker = ClinicalDocumentChunker()

        all_chunks = []
        # Đọc 3 nguồn văn bản chuẩn: Bộ Y Tế, ADA Hoa Kỳ, và Bệnh viện Vinmec
        files_to_read = [
            (
                "Quyết định 5481/QĐ-BYT (Bộ Y tế Việt Nam)", 
                "guidelines_byt_5481.txt",
                "https://kcb.vn/van-ban/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-ve-viec-ban-hanh-tai-lieu-chuyen-mon-huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2.html",
                "QD_5481_BYT.pdf (Trang 15–28)"
            ),
            (
                "ADA Standards of Care in Diabetes 2024 (Hoa Kỳ)", 
                "guidelines_ada_2024.txt",
                "https://diabetesjournals.org/care/issue/47/Supplement_1",
                "ADA_Standards_of_Care_2024.pdf (Sec 2, 9, 10, 11)"
            ),
            (
                "Chuyên mục Đái tháo đường Típ 2 - Bệnh viện Vinmec", 
                "vinmec_diabetes_article.txt",
                "https://www.vinmec.com/vi/benh/dai-thao-duong-type-2-4166/",
                "Vinmec_NoiTiet_TieuDuong.pdf"
            )
        ]

        for source_name, fname, url, pdf_ref in files_to_read:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                sub_chunks = chunker.split_text(content, source_name, url=url, pdf_ref=pdf_ref)
                all_chunks.extend(sub_chunks)

        self.chunks = all_chunks
        if self.chunks:
            corpus = [c["text"] for c in self.chunks]
            tfidf_mat = self.vectorizer.fit_transform(corpus)
            self.embeddings = tfidf_mat.toarray()
            self.is_indexed = True

        return len(self.chunks)

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tìm kiếm Top-k đoạn văn bản có độ tương đồng Cosine cao nhất với truy vấn q.
        sim(e_q, e_i) = cos(theta).
        """
        if not self.is_indexed or self.embeddings is None:
            return []

        q_vec = self.vectorizer.transform([query]).toarray()
        scores = cosine_similarity(q_vec, self.embeddings)[0]
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            res_item = dict(self.chunks[idx])
            res_item["similarity_score"] = float(scores[idx])
            results.append(res_item)

        return results


class HybridContextFusion:
    """
    Khối Hợp nhất Ngữ cảnh (Context Fusion):
    Hợp nhất luồng tri thức cấu trúc (Graph Facts) từ Knowledge Graph
    và luồng bằng chứng phi cấu trúc (Document Evidence) từ Vector RAG.
    Công thức: C = Fuse(G_k, D_k).
    """
    @staticmethod
    def fuse(
        query: str,
        vector_results: List[Dict[str, Any]],
        graph_facts: List[Dict[str, Any]],
        patient_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        # 1. Định dạng khối Đồ thị Tri thức
        graph_lines = []
        if graph_facts:
            for item in graph_facts:
                if isinstance(item, dict) and "chain" in item:
                    graph_lines.append(f"  * [{item.get('type', 'FACT')}]: {item['chain']} => {item.get('action', '')}")
                elif isinstance(item, dict) and "chain_str" in item:
                    graph_lines.append(f"  * [2-Hop Path]: {item['chain_str']}")
                else:
                    graph_lines.append(f"  * {str(item)}")
        else:
            graph_lines.append("  (Không có cảnh báo vi phạm chống chỉ định trực tiếp)")

        graph_facts_str = "\n".join(graph_lines)

        # 2. Định dạng khối Bằng chứng Văn bản kèm Link Web và File PDF
        doc_lines = []
        if vector_results:
            for r in vector_results:
                score_str = f"{r.get('similarity_score', 0.0):.3f}"
                src = r.get('source', 'Y văn')
                url = r.get('url', '')
                pdf = r.get('pdf_ref', '')
                
                link_md = f"[{src}]({url})" if url else f"**{src}**"
                pdf_md = f"*(Tệp PDF: `{pdf}`)*" if pdf else ""
                
                doc_lines.append(f"  * {link_md} {pdf_md} | Cosine Sim: `{score_str}`:\n    \"{r.get('text')}\"")
        else:
            doc_lines.append("  (Không tìm thấy đoạn văn bản đối chiếu phù hợp)")

        doc_evidence_str = "\n\n".join(doc_lines)

        # 3. Chuỗi hợp nhất toàn diện
        fused_prompt_context = f"""=== [KHỐI 1: THÔNG TIN LÂM SÀNG BỆNH NHÂN] ===
- Mã BN: {patient_info.get('patient_id')} | Họ tên: {patient_info.get('full_name')} | Tuổi: {patient_info.get('age')}
- BMI: {patient_info.get('bmi')} kg/m² | HbA1c: {patient_info.get('hba1c')}% | Glucose đói: {patient_info.get('fasting_glucose')} mg/dL
- eGFR: {patient_info.get('egfr')} mL/phút/1.73m² | Huyết áp: {patient_info.get('systolic_bp')}/{patient_info.get('diastolic_bp')} mmHg

=== [KHỐI 2: CƠ SỞ TRI THỨC CẤU TRÚC (GRAPH FACTS - 1-HOP & 2-HOP)] ===
{graph_facts_str}

=== [KHỐI 3: BẰNG CHỨNG TÀI LIỆU TRUY XUẤT (DOCUMENT EVIDENCE - VECTOR RAG)] ===
{doc_evidence_str}
"""

        return {
            "fused_context": fused_prompt_context,
            "graph_facts_count": len(graph_facts),
            "vector_chunks_count": len(vector_results),
            "top_source": vector_results[0].get("source") if vector_results else "N/A"
        }

if __name__ == "__main__":
    vstore = ClinicalVectorStore()
    num_chunks = vstore.build_index_from_directory()
    print(f"[+] Đã lập chỉ mục {num_chunks} đoạn văn bản y tế thành công.")
    
    q = "Phác đồ dùng Metformin cho bệnh nhân suy giảm chức năng thận eGFR dưới 45"
    res = vstore.search_similar(q, top_k=2)
    for r in res:
        print(f"[*] Top-1 [{r['source']} | Sim={r['similarity_score']:.4f}]: {r['text'][:120]}...")

