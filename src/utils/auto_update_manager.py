import csv
import os
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from automation_pipeline import main_pipeline
from wiki_browser_fallback import fetch_strategems_via_browser


WIKI_URL = "https://helldivers.wiki.gg/wiki/Stratagems"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://helldivers.wiki.gg/",
    "Cache-Control": "no-cache",
}
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
VERSION_FILE = os.path.join(PROJECT_ROOT, "version.txt")
CSV_FILE = os.path.join(PROJECT_ROOT, "src", "strategems.csv")
IMG_DIR = os.path.join(PROJECT_ROOT, "img")


def parse_http_datetime(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
    except ValueError:
        return None


def get_local_last_update():
    """Read the most recent local update timestamp from version.txt."""
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r", encoding="utf-8") as file_obj:
            try:
                return datetime.fromisoformat(file_obj.read().strip())
            except ValueError:
                return None
    return None


def update_local_version(dt):
    """Write the latest update timestamp to version.txt."""
    with open(VERSION_FILE, "w", encoding="utf-8") as file_obj:
        file_obj.write(dt.isoformat())


def load_local_strategem_entries():
    """Load local CSV entries and report local CSV issues."""
    issues = []

    if not os.path.isfile(CSV_FILE):
        issues.append(f"Missing CSV file: {CSV_FILE}")
        return set(), set(), issues

    try:
        with open(CSV_FILE, "r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            required_fields = {"Index", "Name", "Code"}
            missing_fields = required_fields - set(reader.fieldnames or [])
            if missing_fields:
                issues.append(
                    "CSV is missing required column(s): "
                    + ", ".join(sorted(missing_fields))
                )
                return set(), set(), issues

            entries = set()
            codes = set()
            for row in reader:
                name = (row.get("Name") or "").strip()
                code = (row.get("Code") or "").strip()
                if not name or not code:
                    continue
                entries.add((name, code))
                codes.add(code)

        if not codes:
            issues.append("CSV exists but contains no strategem entries.")

        return entries, codes, issues
    except Exception as error:
        issues.append(f"Failed to read CSV: {error}")
        return set(), set(), issues


def extract_entries_from_html(html_text):
    """Extract strategem entries from raw wiki HTML."""
    direction_map = {"LEFT": "1", "UP": "2", "RIGHT": "3", "DOWN": "4"}
    soup = BeautifulSoup(html_text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    if not tables:
        raise RuntimeError("No wiki tables with class 'wikitable' were found.")

    entries = set()
    for table in tables:
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            code_cell = None
            name_cell = None
            for index, cell in enumerate(cells):
                if cell.find("span", class_="Stratagemcodeicon"):
                    code_cell = cell
                    if index > 0:
                        name_cell = cells[index - 1]
                    break

            if not code_cell or not name_cell:
                continue

            name = name_cell.get_text(strip=True)
            arrow_imgs = code_cell.find_all("img")
            if (
                not name
                or not arrow_imgs
                or len(name) > 40
                or any(token in name for token in ["Cooldown", "Cost", "Uses"])
            ):
                continue

            code_parts = []
            for img in arrow_imgs:
                match = re.search(r"Arrow\s(\w+)", img.get("alt", ""), re.I)
                if match:
                    mapped = direction_map.get(match.group(1).upper())
                    if mapped:
                        code_parts.append(mapped)

            code = "".join(code_parts)
            if name and code:
                entries.add((name, code))

    if not entries:
        raise RuntimeError("Wiki responded, but no strategem entries could be parsed.")

    return entries


def fetch_remote_snapshot():
    """Fetch remote wiki data, falling back to a real browser if requests is blocked."""
    try:
        response = requests.get(
            WIKI_URL,
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        entries = extract_entries_from_html(response.text)
        return {
            "source": "requests",
            "last_modified": parse_http_datetime(response.headers.get("Last-Modified")),
            "entries": entries,
            "warnings": [],
            "errors": [],
        }
    except Exception as request_error:
        try:
            payload = fetch_strategems_via_browser()
            entries = {
                ((item.get("Name") or "").strip(), (item.get("Code") or "").strip())
                for item in payload.get("entries", [])
                if (item.get("Name") or "").strip() and (item.get("Code") or "").strip()
            }
            if not entries:
                raise RuntimeError("Playwright fallback loaded the page but returned no strategem entries.")

            return {
                "source": "playwright",
                "last_modified": parse_http_datetime(payload.get("lastModified")),
                "entries": entries,
                "warnings": [f"Requests fetch was blocked or failed: {request_error}"],
                "errors": [],
            }
        except Exception as browser_error:
            return {
                "source": None,
                "last_modified": None,
                "entries": set(),
                "warnings": [],
                "errors": [
                    f"Requests fetch failed: {request_error}",
                    f"Playwright fallback failed: {browser_error}",
                ],
            }


def verify_local_assets():
    """Verify the local CSV and icon assets used by the app."""
    issues = []
    entries, codes, csv_issues = load_local_strategem_entries()
    issues.extend(csv_issues)

    if not os.path.isdir(IMG_DIR):
        issues.append(f"Missing image directory: {IMG_DIR}")
        return {"issues": issues, "entries": entries, "codes": codes, "icon_count": 0}

    icon_files = [
        file_name for file_name in os.listdir(IMG_DIR)
        if file_name.lower().endswith(".png")
    ]

    if not icon_files:
        issues.append(f"No icon files found in: {IMG_DIR}")

    if codes:
        missing_icons = [
            code for code in sorted(codes)
            if not os.path.isfile(os.path.join(IMG_DIR, f"{code}.png"))
        ]
        if missing_icons:
            preview = ", ".join(missing_icons[:10])
            suffix = "..." if len(missing_icons) > 10 else ""
            issues.append(f"Missing {len(missing_icons)} icon file(s): {preview}{suffix}")

    return {
        "issues": issues,
        "entries": entries,
        "codes": codes,
        "icon_count": len(icon_files),
    }


def check_and_update():
    print(f"[{datetime.now()}] Checking for Helldivers 2 Wiki updates...")

    asset_report = verify_local_assets()
    remote_snapshot = fetch_remote_snapshot()
    local_time = get_local_last_update()

    reasons = []

    if asset_report["issues"]:
        print("Local asset verification failed:")
        for issue in asset_report["issues"]:
            print(f"  - {issue}")
        reasons.append("local assets are incomplete")
    else:
        print(
            "Local asset verification passed: "
            f"{len(asset_report['codes'])} code(s), "
            f"{asset_report['icon_count']} icon file(s)."
        )

    for warning in remote_snapshot["warnings"]:
        print(f"Remote fetch warning: {warning}")

    if remote_snapshot["errors"]:
        print("Could not verify remote wiki content:")
        for issue in remote_snapshot["errors"]:
            print(f"  - {issue}")
    else:
        print(
            f"Remote wiki verification loaded {len(remote_snapshot['entries'])} "
            f"strategem entries via {remote_snapshot['source']}."
        )
        print(f"Local version timestamp: {local_time}")
        print(f"Remote last-modified timestamp: {remote_snapshot['last_modified']}")

        if local_time is None:
            reasons.append("local version file is missing or invalid")
        elif remote_snapshot["last_modified"] and remote_snapshot["last_modified"] > local_time:
            reasons.append("wiki has newer data than local version")

        if asset_report["entries"] and asset_report["entries"] != remote_snapshot["entries"]:
            reasons.append("local CSV does not match remote wiki data")

    if reasons:
        print("Starting Automation Pipeline because " + ", ".join(reasons) + ".")
        main_pipeline()
        post_report = verify_local_assets()
        if post_report["issues"]:
            print("Pipeline finished, but local asset verification still failed:")
            for issue in post_report["issues"]:
                print(f"  - {issue}")
            print("Update did not complete successfully.")
            return False

        update_local_version(remote_snapshot["last_modified"] or datetime.now())
        print("Verification and update completed successfully.")
        return True

    if remote_snapshot["errors"]:
        print("Local assets are verified, but remote wiki verification is unavailable.")
        return False

    print("Everything is up to date and local assets are verified.")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if check_and_update() else 1)
