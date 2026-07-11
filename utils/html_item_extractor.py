#!/usr/bin/env python3
"""HTML anchor-based Item extractor for SEC filings.

Usage:
  python utils/html_item_extractor.py \
    --html data/sec_filings_html/HAVA_2026_10K.htm \
    --items 1,1A,3,7,10,11 \
    --out extracted_items.json

This script parses the HTML, finds TOC anchors (e.g. #a_036), maps them to Item numbers,
locates each Item's start anchor in the document, and extracts text until the next Item.
"""

import re
import sys
import json
import argparse
from pathlib import Path
from bs4 import BeautifulSoup

def build_toc_map(soup):
    mapping = {}
    # Find links with hrefs pointing to anchors like #a_036
    for link in soup.select('a[href^="#a_"]'):
        href = link.get('href', '').lstrip('#')
        tr = link.find_parent('tr')
        if tr:
            tds = tr.find_all(['td', 'th'])
            if tds:
                txt = tds[0].get_text(separator=' ').strip()
                m = re.search(r'Item\s+(\d+[A-Z]?)', txt, re.I)
                if m:
                    mapping[m.group(1).upper()] = href
    # fallback: try to parse any link text like 'Item 1.' in siblings

    print(mapping)
    return mapping

def find_start_elem(soup, anchor_id, item_label=None):
    if anchor_id:
        elem = soup.find(id=anchor_id)
        if elem:
            return elem
    if item_label:
        regex = re.compile(r'^\s*Item\s+%s\.' % re.escape(item_label), re.I)
        for tag in soup.find_all(text=regex):
            return tag.parent
    return None

def extract_until_next(start_elem, anchor_ids_set, current_anchor_id):
    texts = []
    # iterate through elements after the start element
    for node in start_elem.next_elements:
        # stop if we hit another anchor id
        if getattr(node, 'get', None):
            node_id = node.get('id')
            if node_id and node_id in anchor_ids_set and node_id != current_anchor_id:
                break
        if isinstance(node, str):
            txt = node.strip()
            if txt:
                texts.append(txt)
        else:
            # block-level heading that looks like 'Item N.' starts
            if node.name in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                t = node.get_text(separator=' ', strip=True)
                if re.match(r'^Item\s+\d+[A-Z]?\.', t, re.I):
                    break
    return '\n\n'.join(texts).strip()

def extract_items(html_path, items):
    html = Path(html_path).read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'lxml')

    toc_map = build_toc_map(soup)
    anchor_ids = set(toc_map.values())

    results = {}
    for item in items:
        key = item.upper()
        anchor = toc_map.get(key)
        start_elem = None
        if anchor:
            start_elem = find_start_elem(soup, anchor, key)
        else:
            # attempt to find by heading text
            start_elem = find_start_elem(soup, None, key)

        if not start_elem:
            results[item] = {
                "item": item,
                "title": f"Item {item}",
                "text": clean_text(text)
            }
            continue

        text = extract_until_next(start_elem, anchor_ids, anchor)

        if not text:
            text = start_elem.get_text(separator=' ', strip=True)

        results[item] = {
            "title": f"Item {item}",
            "text": clean_text(text)
        }

    return results

def extract_until_next(start_elem, anchor_ids_set, current_anchor_id):
    texts = []
    seen = set()

    for node in start_elem.next_elements:

        # Stop at next SEC Item
        if getattr(node, "get", None):
            node_id = node.get("id")
            if node_id and node_id in anchor_ids_set and node_id != current_anchor_id:
                break

        # Only process block elements
        if getattr(node, "name", None) not in {
            "p", "div", "li",
            "h1", "h2", "h3", "h4", "h5", "h6"
        }:
            continue

        # Avoid processing the same tag twice
        if id(node) in seen:
            continue
        seen.add(id(node))

        text = node.get_text(" ", strip=True)

        if not text:
            continue

        # Stop if we accidentally reached the next Item heading
        if re.match(r"^Item\s+\d+[A-Z]?\.", text, re.I):
            break

        texts.append(text)

    return "\n\n".join(texts)

def clean_text(text):
    # Remove page metadata
    text = re.sub(r'Field:.*', '', text)
    # Remove isolated page numbers
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove Page/Sequence markers
    text = re.sub(r'^/?Page\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^/?Sequence\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^PageNo\s*$', '', text, flags=re.MULTILINE)

    # Join wrapped lines
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Collapse spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--html', required=True)
    p.add_argument('--items', required=False, default='1,1A,3,7,10,11', help='Comma-separated list of items')
    p.add_argument('--out', required=False, default='extracted_items.json')
    # p.add_argument('--out', required=False, default='extracted_items.txt')
    args = p.parse_args()

    items = [x.strip() for x in args.items.split(',') if x.strip()]
    res = extract_items(args.html, items)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {len(res)} items to {out_path}')

    # combined_text = ""
    # for section, content in res.items():
    #     combined_text += f"\n\n--- BEGIN {section.upper()} ---\n\n"
    #     combined_text += content

    # out_path.write_text(combined_text, encoding="utf-8")
    # print(f'Wrote {len(res)} items to {out_path}')

if __name__ == '__main__':
    main()