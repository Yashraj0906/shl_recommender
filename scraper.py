"""
SHL Product Catalog Scraper

Scrapes https://www.shl.com/solutions/products/product-catalog/
Scope: Individual Test Solutions only (type=1)
       Skips Pre-packaged Job Solutions (type=2)

Pagination: ?start=X&type=1 where X increments by 12

Output: catalog.json -- the SINGLE SOURCE OF TRUTH for all URLs.
Every URL the agent ever returns must come from this file.

Usage:
    python scraper.py
"""

import json
import time
import re
import requests
from bs4 import BeautifulSoup
from typing import Optional, List, Dict


BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Test type code mapping
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "P": "Personality & Behavior",
    "K": "Knowledge & Skills",
    "B": "Biodata & Situational Judgment",
    "C": "Competency",
    "S": "Simulations",
    "D": "Development & 360",
}


def fetch_page(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a page with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"  [Attempt {attempt + 1}/{retries}] Error fetching {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def parse_duration(text: str) -> Optional[int]:
    """Extract duration in minutes from text."""
    if not text:
        return None
    match = re.search(r'(\d+)\s*(?:min|minute)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def scrape_catalog_page(start: int, type_id: int = 1) -> List[Dict]:
    """
    Scrape a single page of the catalog.
    
    SHL catalog uses: ?start=X&type=1 for Individual Test Solutions
    Each page shows 12 items.
    """
    url = f"{BASE_URL}?start={start}&type={type_id}"
    print(f"\n[PAGE] Scraping: {url}")
    
    soup = fetch_page(url)
    if not soup:
        print(f"  [ERROR] Failed to fetch page")
        return []
    
    products = []
    
    # Find all tables on the page
    tables = soup.find_all("table")
    
    for table in tables:
        rows = table.find_all("tr")
        
        for row in rows:
            cells = row.find_all("td")
            if not cells or len(cells) < 2:
                continue
            
            # Find the product link in the first cell
            link = None
            for cell in cells:
                a_tag = cell.find("a", href=True)
                if a_tag and "/product-catalog/view/" in a_tag.get("href", ""):
                    link = a_tag
                    break
            
            if not link:
                continue
            
            name = link.get_text(strip=True)
            href = link.get("href", "")
            
            if not name or len(name) < 2:
                continue
            
            # Build absolute URL
            if href.startswith("/"):
                product_url = f"https://www.shl.com{href}"
            elif href.startswith("http"):
                product_url = href
            else:
                continue
            
            # Extract test type badges from the row
            # SHL uses letter badges (A, B, K, P, S) in table cells
            test_types = []
            row_text = row.get_text()
            
            # Look for standalone single-letter type indicators
            for cell in cells[1:]:  # Skip the name cell
                cell_text = cell.get_text(strip=True)
                # Check for single or comma-separated type codes
                for char in cell_text.replace(",", " ").split():
                    char = char.strip()
                    if char in TEST_TYPE_MAP:
                        test_types.append(char)
            
            # Check for green dot indicators (remote testing, adaptive)
            remote_testing = False
            adaptive_irt = False
            
            for i, cell in enumerate(cells):
                # Green dots or checkmarks typically in specific columns
                has_indicator = (
                    cell.find("span", class_=re.compile(r"catalogue__circle", re.I)) or
                    cell.find("span", class_=re.compile(r"green|check|yes|dot", re.I)) or
                    cell.find("i", class_=re.compile(r"check|yes", re.I))
                )
                if has_indicator:
                    # Column positions typically: name, remote, adaptive, test_types...
                    if i == 1:
                        remote_testing = True
                    elif i == 2:
                        adaptive_irt = True
            
            test_type = ",".join(test_types) if test_types else "K"
            
            product = {
                "name": name,
                "url": product_url,
                "test_type": test_type,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
            }
            products.append(product)
    
    print(f"  Found {len(products)} products")
    return products


def scrape_product_detail(url: str) -> Dict:
    """Scrape the detail page for a single assessment."""
    detail = {
        "description": "",
        "job_levels": [],
        "duration_minutes": None,
    }
    
    soup = fetch_page(url)
    if not soup:
        return detail
    
    # Strategy 1: meta description tag
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        detail["description"] = meta_desc["content"].strip()
    
    # Strategy 2: look for content paragraphs if meta is empty
    if not detail["description"]:
        # Find the main content area
        content_areas = soup.find_all("div", class_=re.compile(
            r"product|content|description|overview|detail", re.I
        ))
        for area in content_areas:
            paragraphs = area.find_all("p")
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 40:
                    detail["description"] = text
                    break
            if detail["description"]:
                break
    
    # Strategy 3: first substantial paragraph on page
    if not detail["description"]:
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50 and not text.startswith("Cookie"):
                detail["description"] = text
                break
    
    # Extract job levels from page text
    page_text = soup.get_text()
    job_level_patterns = [
        "Entry-Level", "Graduate", "Mid-Professional",
        "Professional", "Manager", "Director", "Executive",
        "Front Line Manager", "Senior Manager"
    ]
    for level in job_level_patterns:
        if level.lower() in page_text.lower():
            detail["job_levels"].append(level)
    
    # Extract duration
    duration_match = re.search(
        r'(?:duration|time|length|approximately|approx)\s*:?\s*(\d+)\s*(?:min|minute)',
        page_text, re.I
    )
    if duration_match:
        detail["duration_minutes"] = int(duration_match.group(1))
    
    return detail


def scrape_individual_tests() -> List[Dict]:
    """
    Scrape ALL Individual Test Solutions (type=1).
    Paginate from start=0 in increments of 12 until no new products found.
    """
    all_products = []
    seen_urls = set()
    
    # type=1 = Individual Test Solutions (up to ~32 pages = ~384 products)
    max_pages = 40  # Safety limit
    
    for page in range(max_pages):
        start = page * 12
        products = scrape_catalog_page(start=start, type_id=1)
        
        new_count = 0
        for product in products:
            if product["url"] not in seen_urls:
                seen_urls.add(product["url"])
                all_products.append(product)
                new_count += 1
        
        print(f"  {new_count} new (total: {len(all_products)})")
        
        if len(products) == 0 or new_count == 0:
            print("  No new products found, stopping pagination.")
            break
        
        time.sleep(1)  # Be respectful to the server
    
    return all_products


def enrich_with_details(catalog: List[Dict]) -> List[Dict]:
    """Fetch detail pages for all products to get descriptions and job levels."""
    print(f"\n[DETAIL] Fetching details for {len(catalog)} products...")
    
    for i, product in enumerate(catalog):
        print(f"  [{i + 1}/{len(catalog)}] {product['name']}")
        details = scrape_product_detail(product["url"])
        product.update(details)
        time.sleep(0.5)  # Rate limiting
    
    return catalog


def main():
    print("=" * 60)
    print("SHL Product Catalog Scraper")
    print("Scope: Individual Test Solutions only (type=1)")
    print("=" * 60)
    
    # Step 1: Scrape catalog listing pages
    catalog = scrape_individual_tests()
    
    # Step 2: Enrich with detail page data
    catalog = enrich_with_details(catalog)
    
    # Step 3: Save
    output_path = "catalog.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Scraped {len(catalog)} assessments -> {output_path}")
    
    # Print summary
    type_counts = {}
    for item in catalog:
        for t in item.get("test_type", "").split(","):
            t = t.strip()
            if t:
                type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\n[STATS] Test type distribution:")
    for code, count in sorted(type_counts.items()):
        label = TEST_TYPE_MAP.get(code, "Unknown")
        print(f"  {code} ({label}): {count}")
    
    # Validate URLs
    print("\n[CHECK] URL validation:")
    broken = [item for item in catalog if not item["url"].startswith("https://www.shl.com")]
    if broken:
        print(f"  [WARN] {len(broken)} items with non-SHL URLs!")
        for item in broken[:5]:
            print(f"    - {item['name']}: {item['url']}")
    else:
        print("  [OK] All URLs are valid SHL catalog URLs")
    
    # Show sample
    print("\n[SAMPLE] First 5 items:")
    for item in catalog[:5]:
        print(f"  - {item['name']} ({item['test_type']}) -> {item['url']}")


if __name__ == "__main__":
    main()
