"""
EACL 2025-2026 Research Corpus Harvester & PDF Downloader (Powered by requests)
Curates 100 recent (2025-2026) research papers directly relevant to:
1. Multi-Agent Collusion, Deception & Persuasion
2. Belief Manipulation & Epistemic Tracking
3. Generative Misinformation & Rumor Verification
4. Causal Reasoning, DAGs & Chain Provenance
5. LLM Safety Guardrails, Epistemic Defenses & Hallucination Mitigation

Downloads all 100 PDFs into eacl_papers_2025_2026/pdfs/
Generates papers_metadata.json and a formatted research README.md.
"""

import os
import re
import sys
import time
import json
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent.parent / "eacl_papers_2025_2026"
PDF_DIR = TARGET_DIR / "pdfs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

SEARCH_TOPICS = [
    {
        "category": "Multi-Agent Collusion & Social Deception",
        "query": "cat:cs.CL AND (all:multi-agent OR all:agent) AND (all:deception OR all:collusion OR all:manipulation OR all:persuasion)"
    },
    {
        "category": "Belief Manipulation & Epistemic Tracking",
        "query": "cat:cs.CL AND (all:belief OR all:epistemic) AND (all:manipulation OR all:tracking OR all:reasoning)"
    },
    {
        "category": "Generative Misinformation & Rumor Verification",
        "query": "cat:cs.CL AND (all:misinformation OR all:disinformation OR all:rumor OR all:fake) AND (all:LLM OR all:language)"
    },
    {
        "category": "Causal Reasoning, DAGs & Chain Provenance",
        "query": "cat:cs.CL AND (all:causal OR all:DAG OR all:provenance) AND (all:reasoning OR all:chain-of-thought)"
    },
    {
        "category": "LLM Cognitive Defenses & Safety Guardrails",
        "query": "cat:cs.CL AND (all:safety OR all:guardrail OR all:jailbreak OR all:defense) AND (all:alignment OR all:persuasion)"
    }
]

def sanitize_filename(title: str, max_len: int = 70) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', '', title)
    cleaned = re.sub(r'\s+', '_', cleaned).strip('_')
    return cleaned[:max_len]

def fetch_arxiv_topic(query_str: str, max_results: int = 25) -> list:
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
            
            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "published": published,
                "abstract": abstract,
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
    print("EACL 2025-2026 LITERATURE HARVESTER (Target: 100 Research Papers)")
    print("=" * 95)
    print(f"Destination: {TARGET_DIR}\n")
    
    all_papers = {}
    
    for item in SEARCH_TOPICS:
        cat = item["category"]
        q = item["query"]
        print(f"[*] Fetching category: '{cat}'...")
        results = fetch_arxiv_topic(q, max_results=28)
        added = 0
        for p in results:
            aid = p["arxiv_id"]
            if aid not in all_papers and len(all_papers) < 100:
                p["category"] = cat
                all_papers[aid] = p
                added += 1
        print(f"    Found {len(results)} matches -> Added {added} unique papers. (Current count: {len(all_papers)})")
        time.sleep(2.0)
        if len(all_papers) >= 100:
            break
            
    # Supplementary query if needed
    if len(all_papers) < 100:
        needed = 100 - len(all_papers)
        print(f"\n[*] Running supplementary search for remaining {needed} papers...")
        supp = [
            ("Hallucination Mitigation & Epistemic Faithfulness", "cat:cs.CL AND all:hallucination AND (all:factuality OR all:faithfulness)"),
            ("Multi-Agent Dialogue & Persuasion", "cat:cs.CL AND all:agent AND (all:debate OR all:persuasion)")
        ]
        for sub_cat, sub_q in supp:
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
    print(f"\n[+] Total papers successfully cataloged: {len(papers_list)}")
    
    # Save metadata
    index_path = TARGET_DIR / "papers_metadata.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(papers_list, f, indent=2)
    print(f"[+] Metadata catalog saved to: {index_path.name}")
    
    # Download PDFs
    print("\n" + "=" * 95)
    print(f"DOWNLOADING {len(papers_list)} RESEARCH PAPERS (PDFs) TO: {PDF_DIR.name}/")
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
    md = f"""# EACL 2025-2026 Research Corpus: Belief Manipulation, Multi-Agent Collusion & Epistemic Defense

This research collection contains **{len(papers_list)} cutting-edge papers (2025–2026)** curated specifically to support writing an **EACL (European Chapter of the Association for Computational Linguistics)** paper on **CognitiveGuard**.

## 📊 Summary by Research Area:
"""
    categories = {}
    for p in papers_list:
        categories.setdefault(p["category"], []).append(p)
        
    for cat, items in categories.items():
        md += f"- **{cat}**: {len(items)} papers\n"
        
    md += "\n---\n\n## 📚 Master Papers Catalog with Abstracts & Local PDF Links\n\n"
    
    global_idx = 1
    for cat, items in categories.items():
        md += f"### 📂 {cat}\n\n"
        for p in items:
            authors_str = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
            pdf_link = p.get('local_pdf_path', '')
            md += f"#### {global_idx}. {p['title']}\n"
            md += f"- **Authors:** {authors_str} | **Date:** {p['published']} | **arXiv:** [{p['arxiv_id']}](https://arxiv.org/abs/{p['arxiv_id']})\n"
            md += f"- **PDF Download:** [{pdf_link}]({pdf_link})\n"
            md += f"- **Abstract:**\n> {p['abstract'][:450]}...\n\n"
            global_idx += 1
            
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"\n[+] Master Research Index written to: {readme_path}")
    print(f"[+] Download Completion: {downloaded_count} succeeded, {failed_count} failed out of {len(papers_list)} total.")
    print("=" * 95)

if __name__ == "__main__":
    main()
