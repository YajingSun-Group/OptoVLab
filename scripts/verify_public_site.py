from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from playwright.async_api import Page, async_playwright, expect


ROOT = Path(__file__).resolve().parents[1]


async def no_horizontal_overflow(page: Page) -> dict[str, int]:
    dimensions = await page.evaluate(
        """() => ({
            viewport: window.innerWidth,
            document: document.documentElement.scrollWidth,
            body: document.body.scrollWidth
        })"""
    )
    assert dimensions["document"] <= dimensions["viewport"] + 1, dimensions
    return dimensions


async def verify(base_url: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "base_url": base_url,
        "console_errors": [],
        "page_errors": [],
        "bad_responses": [],
    }
    executable = os.getenv("CHROME_EXECUTABLE") or None
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=executable,
            args=["--no-sandbox", "--disable-gpu"],
        )
        context = await browser.new_context(viewport={"width": 1440, "height": 1000})
        page = await context.new_page()
        page.on(
            "console",
            lambda message: report["console_errors"].append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        page.on(
            "response",
            lambda response: report["bad_responses"].append(
                {"status": response.status, "url": response.url}
            )
            if response.status >= 400
            else None,
        )

        await page.goto(base_url, wait_until="networkidle")
        await expect(page.get_by_role("heading", name="OptoVLab")).to_be_visible()
        await expect(page.locator(".agent-tab")).to_have_count(3)
        report["desktop_layout"] = await no_horizontal_overflow(page)

        expected_titles = {
            "Data Mining Agent": "Evidence-backed OLED extraction",
            "Device Modeling Agent": "Directed OLED graph and quantile EQE",
            "Experimental Design Agent": "Critic-reviewed closed-loop optimization",
        }
        for tab_name, title in expected_titles.items():
            await page.get_by_role("tab", name=tab_name).click()
            await expect(page.locator("#agent-title")).to_have_text(title)
            await expect(page.locator("#tool-events li")).to_have_count(5)

        figures = page.locator(".figure-button")
        await expect(figures).to_have_count(2)
        await figures.first.click()
        await expect(page.locator("#figure-dialog")).to_be_visible()
        await page.get_by_role("button", name="Close figure").click()
        await page.screenshot(path=output_dir / "desktop.png", full_page=True)

        mobile = await context.new_page()
        await mobile.set_viewport_size({"width": 390, "height": 844})
        await mobile.goto(base_url, wait_until="networkidle")
        report["mobile_layout"] = await no_horizontal_overflow(mobile)
        await expect(mobile.locator(".agent-tab")).to_have_count(3)
        await mobile.screenshot(path=output_dir / "mobile.png", full_page=True)
        await mobile.close()
        await context.close()
        await browser.close()

    if report["console_errors"] or report["page_errors"] or report["bad_responses"]:
        raise RuntimeError("Browser verification captured errors")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the public OptoVLab site.")
    parser.add_argument("--base-url", default="http://127.0.0.1:4173")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runtime/site_verification",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    report = await verify(args.base_url.rstrip("/"), args.output_dir)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
