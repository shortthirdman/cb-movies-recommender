"""FastAPI-backend variant of drive.py. See drive.py for how to run these."""

import sys
from pathlib import Path

from drive import LAUNCH, URL, errors, settle
from playwright.sync_api import sync_playwright

SHOTS = Path("shots")


def main() -> int:
    failures: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(**LAUNCH)
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.goto(URL, wait_until="domcontentloaded")
        settle(page)

        page.get_by_text("FastAPI service", exact=True).click()
        settle(page)

        url_box = page.query_selector("[data-testid='stSidebar'] input[type='text']")
        if url_box is None:
            failures.append("no service URL field appeared after switching backend")
        else:
            url_box.fill("http://127.0.0.1:8600")
            url_box.press("Enter")
            settle(page)

        box = page.query_selector("[data-testid='stChatInput'] textarea")
        box.fill("4 films like Toy Story")
        box.press("Enter")
        settle(page)
        page.screenshot(path=SHOTS / "06-service-backend.png", full_page=True)

        if found := errors(page):
            failures.append(f"service backend raised: {found}")
        body = page.inner_text("body")
        if "4 films similar to" not in body:
            failures.append(f"no result via service backend: {body[-800:]!r}")

        headers = page.eval_on_selector_all(
            "[data-testid='stDataFrame'] [role='columnheader']",
            "els => els.map(e => e.getAttribute('aria-label') || e.innerText)",
        )
        for column in ("Film", "Match", "Country", "Year"):
            if column not in headers:
                failures.append(f"service result missing column {column!r}; saw {headers}")

        # Point at a dead port: the page must explain itself, not crash.
        url_box = page.query_selector("[data-testid='stSidebar'] input[type='text']")
        url_box.fill("http://127.0.0.1:9")
        url_box.press("Enter")
        settle(page)
        box = page.query_selector("[data-testid='stChatInput'] textarea")
        box.fill("movies like Alien")
        box.press("Enter")
        settle(page)
        page.screenshot(path=SHOTS / "07-service-unreachable.png", full_page=True)
        if found := errors(page):
            failures.append(f"unreachable service raised instead of reporting: {found}")
        if "did not answer" not in page.inner_text("body"):
            failures.append("unreachable service was not reported to the user")

        browser.close()

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("service backend checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
