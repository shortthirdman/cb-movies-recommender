"""End-to-end checks that drive the running app in a real browser.

Not part of the pytest suite: these need the app running and Playwright
installed, neither of which the unit tests require.

    pip install playwright && playwright install chromium
    streamlit run app/streamlit_app.py --server.port 8599 &
    python tests/e2e/drive.py            # in-process backend, both pages
    uvicorn app.api:app --port 8600 &
    python tests/e2e/drive_service.py    # the FastAPI backend path

Screenshots land in shots/. Both scripts exit non-zero on the first failure.
"""

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8599"
# Set PLAYWRIGHT_CHROMIUM to use a preinstalled browser instead of a downloaded one.
LAUNCH = (
    {"executable_path": os.environ["PLAYWRIGHT_CHROMIUM"]}
    if os.environ.get("PLAYWRIGHT_CHROMIUM")
    else {}
)
SHOTS = Path("shots")
SHOTS.mkdir(exist_ok=True)


def settle(page, timeout: int = 120_000) -> None:
    """Wait until Streamlit has finished its rerun."""
    page.wait_for_selector("[data-testid='stApp']", timeout=timeout)
    page.wait_for_function(
        "() => !document.querySelector(\"[data-testid='stStatusWidget']\")",
        timeout=timeout,
    )
    page.wait_for_timeout(1200)


def errors(page) -> list[str]:
    return [
        e.inner_text()[:400]
        for e in page.query_selector_all("[data-testid='stException'], .stException")
    ]


def main() -> int:
    failures: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(**LAUNCH)
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.on("pageerror", lambda e: failures.append(f"pageerror: {e}"))

        # --- chat page -----------------------------------------------------
        page.goto(URL, wait_until="domcontentloaded")
        settle(page)
        page.screenshot(path=SHOTS / "01-chat-initial.png", full_page=True)
        if found := errors(page):
            failures.append(f"chat initial: {found}")

        body = page.inner_text("body")
        for expected in ("Ask for a recommendation", "Data profiling", "films ·"):
            if expected not in body:
                failures.append(f"chat page missing text: {expected!r}")

        # Type a real request through the chat box.
        box = page.query_selector("[data-testid='stChatInput'] textarea")
        if box is None:
            failures.append("no chat input found")
        else:
            box.fill("recommend 6 movies like Toy Story")
            box.press("Enter")
            settle(page)
            page.screenshot(path=SHOTS / "02-chat-result.png", full_page=True)
            if found := errors(page):
                failures.append(f"chat result: {found}")
            body = page.inner_text("body")
            if "6 films similar to" not in body:
                failures.append(f"no result header in body: {body[:600]!r}")
            # st.dataframe paints into a canvas, so the headers are not in the
            # DOM text. Read them off the grid's accessibility tree instead.
            grid = page.query_selector("[data-testid='stDataFrame']")
            if grid is None:
                failures.append("no result table rendered")
            else:
                headers = page.eval_on_selector_all(
                    "[data-testid='stDataFrame'] [role='columnheader']",
                    "els => els.map(e => e.getAttribute('aria-label') || e.innerText)",
                )
                for column in ("Film", "Match", "Country", "Year"):
                    if column not in headers:
                        failures.append(f"table missing column {column!r}; saw {headers}")
                rows = page.eval_on_selector(
                    "[data-testid='stDataFrame'] [role='grid']",
                    "e => Number(e.getAttribute('aria-rowcount'))",
                )
                if rows != 7:  # 6 results plus the header row
                    failures.append(f"expected 6 result rows, grid reports {rows}")

        # An unknown title must be handled, not crash.
        box = page.query_selector("[data-testid='stChatInput'] textarea")
        box.fill("movies like Zzzqx Nonexistent Film")
        box.press("Enter")
        settle(page)
        if "could not find that film" not in page.inner_text("body"):
            failures.append("unknown title not handled gracefully")
        page.screenshot(path=SHOTS / "03-chat-unknown.png", full_page=True)

        # --- profiling page ------------------------------------------------
        page.goto(f"{URL}/profiling", wait_until="domcontentloaded")
        settle(page)
        page.screenshot(path=SHOTS / "04-profile-controls.png", full_page=True)
        if found := errors(page):
            failures.append(f"profile initial: {found}")
        body = page.inner_text("body")
        for expected in ("Data profiling", "Columns", "Rows sampled", "Minimal mode"):
            if expected not in body:
                failures.append(f"profile page missing control: {expected!r}")

        generate = page.get_by_role("button", name="Generate report")
        generate.click()
        settle(page, timeout=300_000)
        page.wait_for_timeout(6000)
        page.screenshot(path=SHOTS / "05-profile-report.png", full_page=False)
        if found := errors(page):
            failures.append(f"profile report: {found}")

        # The report renders inside the component's iframe.
        frames = [f for f in page.frames if f != page.main_frame]
        if not frames:
            failures.append("no component iframe: report did not render")
        else:
            texts = []
            for frame in frames:
                try:
                    texts.append(frame.inner_text("body")[:4000])
                except Exception:  # frame may be detached mid-read
                    continue
            joined = " ".join(texts)
            for expected in ("Overview", "Variables"):
                if expected not in joined:
                    failures.append(f"report iframe missing {expected!r}: {joined[:300]!r}")

        if "Download report as HTML" not in page.inner_text("body"):
            failures.append("download button missing")

        browser.close()

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
