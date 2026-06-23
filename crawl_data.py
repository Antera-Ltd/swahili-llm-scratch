"""
Web Crawler for Real Data Collection On Web
Author: Shadrackovsky
"""

import os
import time
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


OUTPUT_FOLDER = "./Crawled_Data"
MAX_PAGES_PER_SITE = 80
DELAY_BETWEEN_REQUESTS = (2, 4)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


SWAHILI_SOURCES = [
    ("BBC_Swahili", "https://www.bbc.com/swahili"),
    ("Mwananchi", "https://www.mwananchi.co.tz"),
    ("Swahili_Wikipedia", "https://sw.wikipedia.org"),
    ("Habari_Leo", "https://habarileo.co.tz"),
    ("Kiswahili_Tuko", "https://kiswahili.tuko.co.ke"),
    ("IPP_Media_Sw", "https://www.ippmedia.com/sw"),
    ("Tamisemi_Sw", "https://www.tamisemi.go.tz/sw"),
    ("MoE_Tanzania_Sw", "https://www.moe.go.tz/sw"),
    ("Jamiiforums", "https://jamiiforums.com"),
    ("AllAfrica_Sw", "https://sw.allafrica.com")
]

ENGLISH_SOURCES = [
    ("The_Citizen", "https://www.thecitizen.co.tz"),
    ("Daily_News_Tz", "https://dailynews.co.tz"),
    ("BBC_News_Africa", "https://www.bbc.com/news/world/africa"),
    ("AllAfrica_Tz", "https://allafrica.com/tanzania"),
    ("IPP_Media_En", "https://www.ippmedia.com/en"),
    ("Africa_Review", "https://www.africareview.com"),
    ("DW_Africa", "https://www.dw.com/en/africa"),
    ("The_Conversation_Africa", "https://theconversation.com/africa"),
    ("Tourism_Tz", "https://tanzaniatourism.go.tz/en"),
    ("Tanzania_Wikipedia", "https://en.wikipedia.org/wiki/Tanzania")
]

ALL_SOURCES = SWAHILI_SOURCES + ENGLISH_SOURCES


def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < 25:
            continue
        line = "".join(ch for ch in line if ch.isalnum() or ch in " .,?!'\"%()-:;")
        cleaned.append(line)
    return "\n\n".join(cleaned)


def get_page_links(soup, base_url):
    links = set()
    base_domain = base_url.split("/")[2]
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/"):
            href = base_url.rstrip("/") + href
        elif not href.startswith("http"):
            continue
        if base_domain in href:
            links.add(href)
    return list(links)


def crawl_site(name, start_url):
    visited = set()
    to_visit = [start_url]
    content = []

    while to_visit and len(visited) < MAX_PAGES_PER_SITE:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(url, headers=headers, timeout=12)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")

            main_content = soup.find("article") or soup.find("main") or soup.body
            if main_content:
                text = clean_text(main_content.get_text(separator=" ", strip=True))
                if len(text) > 100:
                    content.append(text)

            new_links = get_page_links(soup, start_url)
            for link in new_links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

            visited.add(url)
            time.sleep(random.uniform(*DELAY_BETWEEN_REQUESTS))

        except Exception:
            continue

    full_text = "\n\n".join(content)
    out_path = os.path.join(OUTPUT_FOLDER, f"{name}_Crawled_Cleaned.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved: {out_path} , Size: {len(full_text)/1024/1024:.2f} MB")


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Starting.. Saving to {OUTPUT_FOLDER}\n")

    for site_name, site_url in tqdm(ALL_SOURCES, desc="Processing sites"):
        print(f"\n Crawling: {site_name} ")
        crawl_site(site_name, site_url)

    print("\nWork's done. All files saved to ./Crawled_Data")