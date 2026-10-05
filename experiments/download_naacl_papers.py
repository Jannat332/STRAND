"""
NAACL Research Corpus Harvester & PDF Downloader
Curates and downloads 100 NAACL (North American Chapter of the ACL) papers
directly relevant to CognitiveGuard:
1. Multi-Agent Systems, Collusion, Deception & Persuasion in NAACL
2. Belief Tracking, Stance & Epistemic Reasoning in NAACL
3. Misinformation, Rumors & Fact Verification in NAACL
4. Causal Inference, Reasoning DAGs & Explainability in NAACL
5. Hallucination Mitigation, Epistemic Calibration & LLM Safety in NAACL

Downloads all 100 PDFs into naacl_papers/pdfs/
Generates papers_metadata.json and an annotated markdown README.md.
"""

import os
import re
import sys
import time
import json
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent.parent / "naacl_papers"
PDF_DIR = TARGET_DIR / "pdfs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Thematic queries specifically targeting NAACL papers
SEARCH_TOPICS = [
    {
        "category": "Misinformation, Rumors & Fact Verification (NAACL)",
        "query": "cat:cs.CL AND all:NAACL AND (all:misinformation OR all:rumor OR all:fake OR all:fact OR all:verification)"
    },
    {
        "category": "Belief Tracking, Persuasion & Stance Detection (NAACL)",
        "query": "cat:cs.CL AND all:NAACL AND (all:belief OR all:persuasion OR all:stance OR all:epistemic OR all:opinion)"
    },
    {
        "category": "Multi-Agent Systems, Dialogue & Social Interaction (NAACL)",
        "query": "cat:cs.CL AND all:NAACL AND (all:agent OR all:dialogue OR all:collusion OR all:social OR all:debate)"
    },
    {
        "category": "Causal Inference, Reasoning DAGs & Faithfulness (NAACL)",
        "query": "cat:cs.CL AND all:NAACL AND (all:causal OR all:reasoning OR all:graph OR all:provenance OR all:explainability)"
    },
    {
        "category": "Hallucination Mitigation, Calibration & LLM Safety (NAACL)",
        "query": "cat:cs.CL AND all:NAACL AND (all:safety OR all:defense OR all:jailbreak OR all:hallucination OR all:calibration)"
    }
]

def sanitize_filename(title: str, max_len: int = 70) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', '', title)
    cleaned = re.sub(r'\s+', '_', cleaned).strip('_')
    return cleaned[:max_len]

def fetch_arxiv_topic(query_str: str, max_results: int = 30) -> list:
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query_str,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=25)
        if resp.status_code != 200:
            print(f"  [Error] Status {resp.status_code}")
            return []
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            published = entry.find("atom:published", ns).text.strip()[:10]
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            
            raw_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
            arxiv_id = raw_id.split("v")[0] if "v" in raw_id else raw_id
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Check comment for NAACL mention
            comment = ""
            comment_elem = entry.find("{http://arxiv.org/schemas/atom}comment")
            if comment_elem is not None and comment_elem.text:
                comment = comment_elem.text.strip()
            
            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "published": published,
                "abstract": abstract,
                "comment": comment,
                "pdf_url": pdf_url
            })
        return papers
    except Exception as e:
        print(f"  [Exception] {e}")
        return []

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 95)
    print("NAACL RESEARCH CORPUS HARVESTER (Target: 100 Papers)")
    print("=" * 95)
    print(f"Destination Folder: {TARGET_DIR}\n")
    
    all_papers = {}
    
    for item in SEARCH_TOPICS:
        cat = item["category"]
        q = item["query"]
        print(f"[*] Fetching category: '{cat}'...")
        results = fetch_arxiv_topic(q, max_results=35)
        added = 0
        for p in results:
            aid = p["arxiv_id"]
            if aid not in all_papers and len(all_papers) < 100:
                p["category"] = cat
                all_papers[aid] = p
                added += 1
        print(f"    Found {len(results)} matches -> Added {added} unique NAACL papers. (Total count: {len(all_papers)})")
        time.sleep(2.0)
        if len(all_papers) >= 100:
            break
            
    # Supplementary query across NAACL cs.CL papers to reach exactly 100
    if len(all_papers) < 100:
        needed = 100 - len(all_papers)
        print(f"\n[*] Running supplementary search across NAACL NLP papers (needed: {needed})...")
        supp_queries = [
            ("Language Model Reasoning & Grounding (NAACL)", "cat:cs.CL AND all:NAACL AND (all:grounding OR all:logic OR all:consistency)"),
            ("General NLP Safety & Robustness (NAACL)", "cat:cs.CL AND all:NAACL AND (all:robustness OR all:adversarial OR all:evaluation)")
        ]
        for sub_cat, sub_q in supp_queries:
            if len(all_papers) >= 100:
                break
            results = fetch_arxiv_topic(sub_q, max_results=needed + 10)
            for p in results:
                aid = p["arxiv_id"]
                if aid not in all_papers and len(all_papers) < 100:
                    p["category"] = sub_cat
                    all_papers[aid] = p
            time.sleep(2.0)

    papers_list = list(all_papers.values())
    print(f"\n[+] Total NAACL papers curated: {len(papers_list)}")
    
    # Save metadata index
    index_path = TARGET_DIR / "papers_metadata.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(papers_list, f, indent=2)
    print(f"[+] Metadata catalog written to: {index_path.name}")
    
    # Download PDFs
    print("\n" + "=" * 95)
    print(f"DOWNLOADING {len(papers_list)} NAACL RESEARCH PAPERS (PDFs) TO: {PDF_DIR.name}/")
    print("=" * 95)
    
    downloaded_count = 0
    failed_count = 0
    
    for i, p in enumerate(papers_list, 1):
        clean_name = f"{i:03d}_{sanitize_filename(p['title'])}.pdf"
        file_path = PDF_DIR / clean_name
        p["local_pdf_path"] = f"pdfs/{clean_name}"
        
        if file_path.exists() and file_path.stat().st_size > 10000:
            sz_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
            print(f"[{i:03d}/{len(papers_list)}] [Cached] {p['title'][:60]}... ({sz_mb} MB)")
            downloaded_count += 1
            continue
            
        print(f"[{i:03d}/{len(papers_list)}] Downloading: {p['title'][:60]}... ", end="", flush=True)
        try:
            r = requests.get(p["pdf_url"], headers=HEADERS, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(file_path, "wb") as out_f:
                    out_f.write(r.content)
                sz_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
                print(f"DONE ({sz_mb} MB)")
                downloaded_count += 1
            else:
                print(f"FAILED (Status {r.status_code})")
                failed_count += 1
            time.sleep(1.0)
        except Exception as err:
            print(f"FAILED ({err})")
            failed_count += 1
            
    # Update metadata with final local paths
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(papers_list, f, indent=2)
        
    # Generate Categorized Research README
    readme_path = TARGET_DIR / "README.md"
    md = f"""# NAACL Research Corpus: Belief Manipulation, Causal Reasoning & NLP Safety

This research collection contains **{len(papers_list)} NAACL (North American Chapter of the Association for Computational Linguistics)** papers specifically curated to support drafting our **NAACL manuscript** on **CognitiveGuard**.

## 📊 Summary by Research Area:
"""
    categories = {}
    for p in papers_list:
        categories.setdefault(p["category"], []).append(p)
        
    for cat, items in categories.items():
        md += f"- **{cat}**: {len(items)} papers\n"
        
    md += "\n---\n\n## 📚 Master NAACL Papers Catalog with Abstracts & Local PDF Links\n\n"
    
    global_idx = 1
    for cat, items in categories.items():
        md += f"### 📂 {cat}\n\n"
        for p in items:
            authors_str = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
            pdf_link = p.get('local_pdf_path', '')
            venue_info = f" | **Venue Note:** `{p['comment']}`" if p.get('comment') else ""
            md += f"#### {global_idx}. {p['title']}\n"
            md += f"- **Authors:** {authors_str} | **Date:** {p['published']} | **arXiv:** [{p['arxiv_id']}](https://arxiv.org/abs/{p['arxiv_id']}){venue_info}\n"
            md += f"- **PDF Download:** [{pdf_link}]({pdf_link})\n"
            md += f"- **Abstract:**\n> {p['abstract'][:450]}...\n\n"
            global_idx += 1
            
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"\n[+] Master NAACL Research Index written to: {readme_path}")
    print(f"[+] Download Completion: {downloaded_count} succeeded, {failed_count} failed out of {len(papers_list)} total.")
    print("=" * 95)

if __name__ == "__main__":
    main()
