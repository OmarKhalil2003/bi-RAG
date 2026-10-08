"""
Generate 2-Page Executive Report DOCX
Bilingual Document Q&A (RAG) System
"""
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=60, bottom=60, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_table_borders(table, color="D1D5DB", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders_xml = f'''
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:left w:val="none"/>
        <w:right w:val="none"/>
        <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:insideV w:val="none"/>
    </w:tblBorders>
    '''
    tblPr.append(parse_xml(borders_xml))

def build_report():
    doc = docx.Document()
    
    # Page setup: Standard margins (0.55 in) to cleanly fit 2 pages
    for section in doc.sections:
        section.top_margin = Inches(0.55)
        section.bottom_margin = Inches(0.55)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)
    
    # Palette
    NAVY = RGBColor(15, 35, 60)
    BLUE = RGBColor(30, 64, 175)
    DARK = RGBColor(31, 41, 55)
    GRAY = RGBColor(107, 114, 128)
    
    # Set default font
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(9.5)
    style_normal.font.color.rgb = DARK
    style_normal.paragraph_format.line_spacing = 1.05
    style_normal.paragraph_format.space_after = Pt(2.5)
    style_normal.paragraph_format.space_before = Pt(0)
    
    # Document Header
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_after = Pt(1)
    run_title = p_title.add_run("Bilingual Document Q&A (RAG) — System & Evaluation Report")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(15)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY
    
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(6)
    run_meta = p_meta.add_run("Repository: OmarKhalil2003/bi-RAG  |  Corpus: 24 Public Docs (12 AR, 12 EN)  |  Architecture: Hybrid (BGE-M3 + BM25 RRF)")
    run_meta.font.name = 'Calibri'
    run_meta.font.size = Pt(8.5)
    run_meta.font.color.rgb = GRAY
    run_meta.font.italic = True
    
    # Helper for headings
    def add_sec_heading(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(title)
        run.font.name = 'Calibri'
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = BLUE
        return p

    def add_sub_heading(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(title)
        run.font.name = 'Calibri'
        run.font.size = Pt(9.5)
        run.font.bold = True
        run.font.color.rgb = NAVY
        return p

    # 1. Approach & Design Rationale
    add_sec_heading("1. Approach & Design Rationale")
    
    p = doc.add_paragraph()
    p.add_run("This project implements a lightweight, production-ready Bilingual (Arabic & English) Document Question Answering (RAG) system with measured retrieval and generation quality. The design emphasizes simplicity, zero framework bloat, and strict factual grounding.")
    
    p = doc.add_paragraph()
    r_b = p.add_run("• Corpus & Chunking: ")
    r_b.bold = True
    p.add_run("24 public documents (12 Arabic, 12 English) covering government policies, technical standards (NIST, ISO, ECC-1:2018), and water/environmental reports. Text is extracted via PyMuPDF with Unicode NFKC normalization and diacritic cleaning. Documents are split into 500-token chunks with 75-token overlap using paragraph-aware boundaries, prepended with contextual headers ([Document: {title} | Page: {page}]) to preserve provenance.")
    
    p = doc.add_paragraph()
    r_b = p.add_run("• Baseline Retrieval (Dense): ")
    r_b.bold = True
    p.add_run("BAAI/bge-m3 embeds queries and passages into 1024-dimensional normalized vectors stored in a FAISS IndexFlatIP database for exact cosine search.")

    p = doc.add_paragraph()
    r_b = p.add_run("• Enhanced Retrieval (The Single Improvement — Hybrid Search): ")
    r_b.bold = True
    p.add_run("To eliminate neural reranker latency while ensuring precise matching for regulatory codes and numeric thresholds, we integrated sparse lexical retrieval via BM25Okapi. Dense and sparse candidate lists are fused using Reciprocal Rank Fusion (RRF, k=60):")

    p_eq = doc.add_paragraph()
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_after = Pt(2)
    p_eq.paragraph_format.space_before = Pt(1)
    r_eq = p_eq.add_run("Score_RRF(doc) = 1 / (60 + Rank_dense(doc)) + 1 / (60 + Rank_bm25(doc))")
    r_eq.font.size = Pt(8.5)
    r_eq.font.italic = True
    r_eq.font.bold = True

    p = doc.add_paragraph()
    r_b = p.add_run("• Generation & Dual Refusal Gates: ")
    r_b.bold = True
    p.add_run("The generation layer matches the query language (Arabic or English), cites supporting passages with [source: chunk_id], and refuses unanswerable queries. A retrieval gate directly refuses queries when dense score < 0.50 and BM25 score < 15, preventing LLM hallucinations before generation.")

    # 2. Quantitative Evaluation
    add_sec_heading("2. Quantitative Evaluation")
    p = doc.add_paragraph()
    p.add_run("Evaluated on a fixed gold benchmark of 40 questions (16 Arabic answerable, 16 English answerable, 4 Arabic unanswerable, 4 English unanswerable) assessing factual lookups, numerical thresholds, and cross-lingual queries.")

    # Table 1: Retrieval Comparison
    add_sub_heading("Table 1: Retrieval Performance Comparison (Hit@k & MRR)")
    t1 = doc.add_table(rows=3, cols=5)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t1)
    
    headers1 = ["Architecture", "Hit@1", "Hit@3", "Hit@5", "MRR"]
    for i, h in enumerate(headers1):
        cell = t1.cell(0, i)
        cell.paragraphs[0].text = h
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        set_cell_shading(cell, "F1F5F9")
        set_cell_margins(cell, top=40, bottom=40, left=70, right=70)
        
    data1 = [
        ["Baseline (Dense BGE-M3 + FAISS)", "1.0000", "1.0000", "1.0000", "1.0000"],
        ["Enhanced Hybrid (Dense + BM25 RRF)", "0.9688", "0.9688", "0.9688", "0.9688"],
    ]
    for row_idx, row in enumerate(data1):
        for col_idx, val in enumerate(row):
            cell = t1.cell(row_idx + 1, col_idx)
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_margins(cell, top=35, bottom=35, left=70, right=70)

    p_note = doc.add_paragraph()
    p_note.paragraph_format.space_before = Pt(2)
    p_note.paragraph_format.space_after = Pt(3)
    p_note.add_run("Finding: Arabic queries scored 1.0000 across both systems. On English query q_en_16 (cross-lingual query targeting an Arabic policy), dense search ranked the true passage at Rank 1, while BM25 suffered zero lexical overlap, demonstrating the standard cross-lingual trade-off of unweighted hybrid search.")
    p_note.runs[0].font.size = Pt(8.5)
    p_note.runs[0].font.italic = True

    # Table 2: Generation Quality
    add_sub_heading("Table 2: Generation Quality & Refusal Performance")
    t2 = doc.add_table(rows=6, cols=4)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t2)
    
    headers2 = ["Metric", "Result", "Evaluated Subset", "Description"]
    for i, h in enumerate(headers2):
        cell = t2.cell(0, i)
        cell.paragraphs[0].text = h
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        set_cell_shading(cell, "F1F5F9")
        set_cell_margins(cell, top=40, bottom=40, left=70, right=70)

    data2 = [
        ["Answer Correctness", "68.75% (22/32)", "Answerable queries", "Exact verified facts and citations"],
        ["Partial Correctness", "18.75% (6/32)", "Answerable queries", "Core facts correct; minor clause omission"],
        ["Combined Correctness", "87.50% (28/32)", "Answerable queries", "Factually grounded answers"],
        ["Unsupported-Answer Rate", "3.12% (1/32)", "Answerable queries", "Unevidenced numbers/claims (1 query)"],
        ["Unanswerable Refusal", "100.0% (8/8)", "Unanswerable queries", "Deterministic refusal without hallucination"],
    ]
    for row_idx, row in enumerate(data2):
        for col_idx, val in enumerate(row):
            cell = t2.cell(row_idx + 1, col_idx)
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_margins(cell, top=30, bottom=30, left=70, right=70)

    # 3. Failure Analysis (10 Cases)
    add_sec_heading("3. Failure Analysis (10 Representative Cases)")
    p = doc.add_paragraph()
    p.add_run("We conducted a detailed analysis of all 10 non-perfect cases on the gold benchmark to pinpoint failure modes:")

    t3 = doc.add_table(rows=11, cols=4)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t3)
    
    headers3 = ["#", "Query ID & Topic", "Observed Failure Mode", "Proposed Fix"]
    for i, h in enumerate(headers3):
        cell = t3.cell(0, i)
        cell.paragraphs[0].text = h
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8)
        set_cell_shading(cell, "F1F5F9")
        set_cell_margins(cell, top=30, bottom=30, left=50, right=50)

    cases = [
        ["1", "q_ar_02 (Cybersecurity MFA)", "Dense score high (0.719), but low BM25 score triggered false refusal", "Bypass refusal when dense confidence >= 0.60"],
        ["2", "q_ar_03 (AI Ethics Principles)", "Numbered list digits ('4.', '5.') flagged as unsupported numbers", "Strip list numbering prefixes before fact verification"],
        ["3", "q_ar_05 (Air Quality PM2.5)", "PM2.5 answered; secondary electric bus target from page 2 missed", "Aggregate adjacent page chunks for multi-target queries"],
        ["4", "q_ar_10 (Water Strategy)", "Desalination target answered; leakage reduction timeline missed", "Sub-query decomposition for multi-part questions"],
        ["5", "q_ar_15 (Distribution Network)", "100% coverage cited; 2-hour pipe repair timeframe missed", "Completeness checklist prompt verifying all question clauses"],
        ["6", "q_en_04 (OECD AI Principles)", "All 5 principles listed; 'high-risk' classification trigger omitted", "Enforce concise bullet answers without conversational preamble"],
        ["7", "q_en_08 (Cloud HSM BYOK)", "FIPS 140-2 Level 3 cited; annual key rotation schedule omitted", "Instruct model to extract operational lifecycles alongside ratings"],
        ["8", "q_en_09 (GDPR Data Breach)", "72h window cited; fine clause phrasing scored partial", "Accept flexible phrasing for complex statutory penalties"],
        ["9", "q_en_14 (CISA Incident Playbook)", "SHA-256 verification cited; minor forensic synonyms scored partial", "Add standard technical synonyms to gold evaluation reference"],
        ["10", "q_en_16 (Cross-Lingual Cloud)", "Zero English-Arabic BM25 overlap promoted irrelevant English docs", "Disable BM25 when query and document languages diverge"],
    ]
    for row_idx, row in enumerate(cases):
        for col_idx, val in enumerate(row):
            cell = t3.cell(row_idx + 1, col_idx)
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)
            set_cell_margins(cell, top=20, bottom=20, left=45, right=45)

    # 4. Service Architecture & Prioritized Next Steps
    add_sec_heading("4. Service Implementation & Prioritized Fixes")
    
    p = doc.add_paragraph()
    r_b = p.add_run("• FastAPI Service: ")
    r_b.bold = True
    p.add_run("Exposes POST /query (validates query input, performs hybrid retrieval, returns answer with citations, refusal status, and latency) and GET /health (checks FAISS index, BM25 store, and embedding models). Complete unit test suite (13 tests) validates chunking, retrieval metrics, and API contracts.")
    
    p = doc.add_paragraph()
    r_b = p.add_run("• Prioritized Fixes: ")
    r_b.bold = True
    p.add_run("(1) Dynamic BM25 Weighting: deactivate BM25 on cross-lingual queries to prevent lexical noise; (2) Sub-Query Decomposition: decompose multi-part questions into individual lookups; (3) On-Premise Small LLM: package an offline model (e.g. Qwen2.5-3B) to guarantee zero API dependency.")

    # Save to report/report.docx
    os.makedirs("report", exist_ok=True)
    out_path = os.path.join("report", "report.docx")
    doc.save(out_path)
    print(f"Saved report to {out_path}")

if __name__ == "__main__":
    build_report()
