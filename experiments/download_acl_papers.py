"""
ACL Research Corpus Harvester & PDF Downloader
Curates and downloads 100 ACL (Association for Computational Linguistics) papers
directly relevant to STRAND / CognitiveGuard:
1. Misinformation, Rumors & Fact Verification (ACL)
2. Belief Tracking, Persuasion & Stance Detection (ACL)
3. Multi-Agent Systems, Dialogue & Social Collusion (ACL)
4. Causal Inference, Reasoning DAGs & Faithfulness (ACL)
5. Hallucination Mitigation, Calibration & LLM Safety (ACL)

Downloads all 100 PDFs into acl_papers/pdfs/
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

TARGET_DIR = Path(__file__).resolve().parent.parent / "acl_papers"
PDF_DIR = TARGET_DIR / "pdfs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Thematic queries specifically targeting ACL papers
SEARCH_TOPICS = [
    {
        "category": "Misinformation, Rumors & Fact Verification (ACL)",
        "query": 'cat:cs.CL AND all:ACL AND (all:misinformation OR all:rumor OR all:fake OR all:fact OR all:verification)'
    },
    {
        "category": "Belief Tracking, Persuasion & Stance Detection (ACL)",
        "query": 'cat:cs.CL AND all:ACL AND (all:belief OR all:persuasion OR all:stance OR all:epistemic OR all:opinion)'
    },
    {
        "category": "Multi-Agent Systems, Dialogue & Social Collusion (ACL)",
        "query": 'cat:cs.CL AND all:ACL AND (all:agent OR all:dialogue OR all:collusion OR all:social OR all:debate)'
    },
    {
        "category": "Causal Inference, Reasoning DAGs & Faithfulness (ACL)",
        "query": 'cat:cs.CL AND all:ACL AND (all:causal OR all:reasoning OR all:graph OR all:provenance OR all:explainability)'
    },
    {
        "category": "Hallucination Mitigation, Calibration & LLM Safety (ACL)",
        "query": 'cat:cs.CL AND all:ACL AND (all:safety OR all:defense OR all:jailbreak OR all:hallucination OR all:calibration)'
    }
]

def sanitize_filename(title: str, max_len: int = 70) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]', '', title)
    cleaned = re.sub(r'\s+', '_', cleaned).strip('_')
    return cleaned[:max_len]

def fetch_arxiv_topic(query_str: str, max_results: int = 40) -> list:
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
            
            # Check comment for ACL mention
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
    print("HARVESTING 100 ACL PAPERS ACROSS 5 THEMATIC PILLARS")
    print("=" * 95)
    
    curated_papers = {}
    
    for topic in SEARCH_TOPICS:
        cat_name = topic["category"]
        query_str = topic["query"]
        print(f"\n[*] Fetching category: '{cat_name}'...")
        
        candidates = fetch_arxiv_topic(query_str, max_results=45)
        print(f"    Found {len(candidates)} candidates from arXiv query.")
        
        added_for_cat = 0
        for p in candidates:
            aid = p["arxiv_id"]
            if aid in curated_papers:
                continue
            
            # Verify ACL relevance
            full_text = (p["title"] + " " + p["abstract"] + " " + p["comment"]).lower()
            if "acl" in full_text:
                p["category"] = cat_name
                curated_papers[aid] = p
                added_for_cat += 1
                if len(curated_papers) >= 100:
                    break
        
        print(f"    Added {added_for_cat} unique ACL papers. (Total count: {len(curated_papers)})")
        if len(curated_papers) >= 100:
            break
        time.sleep(1.0)
        
    # If not yet 100, run a broad top-tier ACL NLP safety & multi-agent query
    if len(curated_papers) < 100:
        needed = 100 - len(curated_papers)
        print(f"\n[*] Running supplementary top-tier query for {needed} remaining ACL papers...")
        supp_candidates = fetch_arxiv_topic(
            'cat:cs.CL AND all:ACL AND (all:reasoning OR all:alignment OR all:truth OR all:agent OR all:safety)',
            max_results=needed + 25
        )
        for p in supp_candidates:
            aid = p["arxiv_id"]
            if aid not in curated_papers:
                full_text = (p["title"] + " " + p["abstract"] + " " + p["comment"]).lower()
                if "acl" in full_text:
                    p["category"] = "Multi-Agent Systems, Dialogue & Social Collusion (ACL)"
                    curated_papers[aid] = p
                    if len(curated_papers) >= 100:
                        break

    papers_list = list(curated_papers.values())[:100]
    print(f"\n[+] Total ACL papers curated: {len(papers_list)}")
    
    # Save metadata index
    index_path = TARGET_DIR / "papers_metadata.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(papers_list, f, indent=2)
    print(f"[+] Metadata catalog written to: {index_path.name}")
    
    # Download PDFs sequentially with safety delay to prevent rate limits
    print("\n" + "=" * 95)
    print(f"DOWNLOADING {len(papers_list)} ACL RESEARCH PAPERS (PDFs) TO: {PDF_DIR.name}/")
    print("=" * 95)
    
    downloaded_count = 0
    failed_count = 0
    
    for idx, paper in enumerate(papers_list, start=1):
        clean_title = sanitize_filename(paper["title"])
        filename = f"{idx:03d}_{clean_title}.pdf"
        dest = PDF_DIR / filename
        paper["local_pdf_path"] = f"pdfs/{filename}"
        
        if dest.exists() and dest.stat().st_size > 15000:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"[{idx:03d}/100] Already cached: {paper['title'][:55]}... ({size_mb:.2f} MB)")
            downloaded_count += 1
            continue
            
        print(f"[{idx:03d}/100] Downloading: {paper['title'][:55]}...", end=" ", flush=True)
        try:
            r = requests.get(paper["pdf_url"], headers=HEADERS, timeout=40)
            if r.status_code == 200 and len(r.content) > 15000:
                with open(dest, "wb") as f:
                    f.write(r.content)
                size_mb = len(r.content) / (1024 * 1024)
                print(f"DONE ({size_mb:.2f} MB)")
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
    md = f"""# ACL Research Corpus: Multi-Agent Collusion, Belief Manipulation & NLP Safety

This research collection contains **{len(papers_list)} ACL (Association for Computational Linguistics)** papers specifically curated to support our research on **STRAND**.

## 📊 Summary by Research Area:
"""
    categories = {}
    for p in papers_list:
        categories.setdefault(p["category"], []).append(p)
        
    for cat, items in categories.items():
        md += f"- **{cat}**: {len(items)} papers\n"
        
    md += "\n---\n\n## 📚 Master ACL Papers Catalog with Abstracts & Local PDF Links\n\n"
    
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
        
    print(f"\n[+] Master ACL Research Index written to: {readme_path}")
    print(f"[+] Download Completion: {downloaded_count} succeeded, {failed_count} failed out of {len(papers_list)} total.")
    print("=" * 95)

if __name__ == "__main__":
    main()
