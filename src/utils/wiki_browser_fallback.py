from pathlib import Path

from playwright.sync_api import sync_playwright


WIKI_URL = "https://helldivers.wiki.gg/wiki/Stratagems"
BROWSER_CANDIDATES = [
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
]


def find_browser_executable():
    for candidate in BROWSER_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def fetch_strategems_via_browser(timeout=120):
    """Fetch strategem data through a real browser when requests is blocked."""
    browser_path = find_browser_executable()
    if not browser_path:
        raise RuntimeError("No supported Edge/Chrome executable was found for browser fallback.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=browser_path,
            headless=False,
        )
        try:
            page = browser.new_page()
            response = page.goto(
                WIKI_URL,
                wait_until="domcontentloaded",
                timeout=timeout * 1000,
            )

            page.wait_for_timeout(3000)
            page.wait_for_selector("table.wikitable", timeout=30000)

            entries = page.evaluate(
                """
                () => {
                  const directionMap = { LEFT: "1", UP: "2", RIGHT: "3", DOWN: "4" };
                  const tables = [...document.querySelectorAll("table.wikitable")];
                  const extracted = [];
                  let globalIndex = 1;

                  for (const table of tables) {
                    for (const row of table.querySelectorAll("tr")) {
                      const cells = [...row.querySelectorAll("td, th")];
                      if (cells.length < 2) continue;

                      let codeCell = null;
                      let nameCell = null;
                      for (let i = 0; i < cells.length; i += 1) {
                        if (cells[i].querySelector("span.Stratagemcodeicon")) {
                          codeCell = cells[i];
                          if (i > 0) nameCell = cells[i - 1];
                          break;
                        }
                      }

                      if (!codeCell || !nameCell) continue;

                      const name = nameCell.textContent.trim();
                      const arrowImgs = [...codeCell.querySelectorAll("img")];
                      if (
                        !name ||
                        !arrowImgs.length ||
                        name.length > 40 ||
                        ["Cooldown", "Cost", "Uses"].some((token) => name.includes(token))
                      ) {
                        continue;
                      }

                      const code = arrowImgs.map((img) => {
                        const match = (img.getAttribute("alt") || "").match(/Arrow\\s(\\w+)/i);
                        return match ? directionMap[match[1].toUpperCase()] || "" : "";
                      }).join("");

                      if (!code) continue;

                      extracted.push({
                        Index: globalIndex,
                        Name: name,
                        Code: code,
                      });
                      globalIndex += 1;
                    }
                  }

                  return extracted;
                }
                """
            )

            headers = response.all_headers() if response else {}
            return {
                "ok": True,
                "title": page.title(),
                "url": page.url,
                "lastModified": headers.get("last-modified"),
                "entryCount": len(entries),
                "entries": entries,
                "source": "playwright",
                "browser": browser_path,
            }
        finally:
            browser.close()
