"""
Face OSINT — find a person by their face.

Four engines, best-available automatically:
  1. LOCAL FACE MATCH (free, always on): deepface (VGG-Face) verifies the
     input face against avatars harvested from the web sweep — a genuine
     1:1 biometric comparison, no API key needed.
  2. Search4Faces API (if SEARCH4FACES_API_KEY set) — real face-match search
     across VK / TikTok / Instagram photo databases.
  3. Reverse-image search launch links (Yandex — the strongest engine for
     faces, Google Lens, Bing Visual) — one click opens each hunt in the
     browser window.
  4. The image itself is described by Ada's vision chain so the spoken brief
     can say WHO she is looking for (approx age, features, context).

Only authorized investigations. Every run is audit-logged by the caller.
"""
import os
import re
import base64
import asyncio
import tempfile
from pathlib import Path

REVERSE_ENGINES = [
    ("Yandex Images (best for faces)",
     "https://yandex.com/images/search?rpt=imageview&url={public_url}"),
    ("Google Lens", "https://lens.google.com/uploadbyurl?url={public_url}"),
    ("Bing Visual", "https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:{public_url}"),
]

# ── LOCAL FACE ENGINE (free, keyless) ────────────────────────────────────────
_LF_AVAILABLE = None


def local_engine_available() -> bool:
    global _LF_AVAILABLE
    if _LF_AVAILABLE is None:
        try:
            import deepface  # noqa: F401
            _LF_AVAILABLE = True
        except Exception:
            _LF_AVAILABLE = False
    return _LF_AVAILABLE


def has_face(image_path: str) -> bool:
    """True if at least one detectable human face is in the image."""
    if not local_engine_available():
        return False
    try:
        from deepface import DeepFace
        res = DeepFace.extract_faces(image_path, detector_backend="opencv",
                                     enforce_detection=False, silent=True)
        return bool(res)
    except Exception:
        return False


async def verify_face(face_path: str, candidate_image_urls: list, max_candidates: int = 12) -> str:
    """Downloads candidate images (avatars from the web sweep) and biometrically
    verifies each against the target face with deepface. Returns a report."""
    if not local_engine_available():
        return "(local face engine unavailable — deepface not installed)"
    urls = [u for u in candidate_image_urls if re.search(r"\.(jpe?g|png|webp)(\?|$)", u, re.I)][:max_candidates]
    if not urls:
        return "(no candidate images found in the sweep to verify against)"

    import httpx
    work = Path(tempfile.mkdtemp(prefix="ada_face_"))
    lines = []
    checked = 0
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0"}) as c:
        for i, url in enumerate(urls):
            try:
                r = await c.get(url)
                if r.status_code != 200 or len(r.content) < 2000:
                    continue
                cand = work / f"cand_{i}.img"
                cand.write_bytes(r.content)
                checked += 1
                res = await asyncio.to_thread(
                    DeepFace_verify, face_path, str(cand))
                if res is None:
                    continue
                distance, verified, model = res
                # distance-tiered verdicts: honest about strength
                if distance < 0.30:
                    tag = "STRONG-MATCH"
                elif distance < 0.45:
                    tag = "MATCH"
                elif distance < 0.70:
                    tag = "POSSIBLE"
                else:
                    tag = "no-match"
                lines.append(f"{tag}  distance={distance:.3f}  model={model}  src={url[:110]}")
            except Exception:
                continue
    if not lines:
        return f"(checked {checked} candidates — none verifiable)"
    matches = sum(1 for l in lines if l.startswith("MATCH"))
    header = f"LOCAL BIOMETRIC VERIFICATION (deepface): {matches} match(es) from {checked} candidates"
    return header + "\n" + "\n".join(lines)


def DeepFace_verify(face_path: str, candidate_path: str):
    """Blocking 1:1 verification. Returns (distance, verified, model) or None
    when either image contains no usable face. Uses the 'skip' backend first
    (avatars/crops are usually already face-centric), then opencv detection."""
    try:
        from deepface import DeepFace
        try:
            res = DeepFace.verify(face_path, candidate_path,
                                  detector_backend="skip", enforce_detection=False,
                                  silent=True)
        except Exception:
            res = DeepFace.verify(face_path, candidate_path,
                                  detector_backend="opencv", enforce_detection=False,
                                  silent=True)
        return float(res["distance"]), bool(res["verified"]), res.get("model", "?")
    except Exception:
        return None


def harvest_image_urls(sweep_text: str) -> list:
    """Pulls candidate avatar/photo URLs out of raw sweep text."""
    pat = re.compile(r"https?://\S+?\.(?:jpe?g|png|webp)(?:\?\S*)?", re.I)
    seen, urls = set(), []
    for m in pat.finditer(sweep_text or ""):
        u = m.group(0)
        if u not in seen and not any(b in u.lower() for b in ("logo", "icon", "sprite", "favicon")):
            seen.add(u)
            urls.append(u)
    return urls


async def harvest_via_image_search(name: str, max_results: int = 15) -> list:
    """THE candidate harvester: DuckDuckGo image search returns real photo
    URLs for the person — news photos, profile shots, event appearances."""
    try:
        from ddgs import DDGS

        def _search():
            with DDGS() as d:
                return [r.get("image", "") for r in d.images(name, max_results=max_results)
                        if r.get("image")]

        return await asyncio.to_thread(_search)
    except Exception as e:
        print(f"[FACE] image-search harvest failed: {str(e)[:80]}")
        return []


# ── REVERSE IMAGE SEARCH (unknown faces — no name needed) ───────────────────
async def yandex_reverse(image_path: str) -> list:
    """Uploads the face to Yandex Images and scrapes 'pages with similar
    images' — the strongest free engine for finding WHO a face belongs to.
    Returns [{title, url}, ...]."""
    import httpx
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    results = []
    try:
        img_bytes = Path(image_path).read_bytes()
        async with httpx.AsyncClient(timeout=40.0, headers=headers, follow_redirects=True) as c:
            # Step 1: upload -> get the cbir page url
            r = await c.post(
                "https://yandex.com/images-apphost/image-download",
                params={"cbird": "111", "images_avatars_size": "orig",
                        "images_avatars_namespace": "images-cbir"},
                files={"file": ("face.jpg", img_bytes, "image/jpeg")})
            if r.status_code != 200:
                return [{"error": f"yandex upload HTTP {r.status_code}"}]
            data = r.json()
            page_url = data.get("url", "")
            if not page_url:
                return [{"error": "yandex gave no result url"}]
            if page_url.startswith("/"):
                page_url = "https://yandex.com" + page_url
            # Step 2: fetch the results page (html with similar-pages blocks)
            r2 = await c.get(page_url)
            html = r2.text
            # CbirSites blocks: <a class="CbirSites-Item-Link" href>title</a>
            for m in re.finditer(
                    r'CbirSites-Item[^>]*?href="([^"]+)"[^>]*>([^<]{4,120})<', html):
                results.append({"url": m.group(1), "title": m.group(2).strip()})
            # fallback: any /search/?text= links Yandex suggests (entity pages)
            for m in re.finditer(r'href="(https://yandex\.com/search/\?text=[^"]+)"[^>]*>([^<]{4,80})<', html):
                results.append({"url": m.group(1), "title": m.group(2).strip()})
    except Exception as e:
        results.append({"error": f"yandex reverse failed: {str(e)[:80]}"})
    return results[:20]


async def bing_visual(image_path: str) -> list:
    """Bing Visual Search upload — returns visually similar pages."""
    import httpx
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    results = []
    try:
        img_bytes = Path(image_path).read_bytes()
        async with httpx.AsyncClient(timeout=40.0, headers=headers, follow_redirects=True) as c:
            r = await c.post(
                "https://www.bing.com/images/search?view=detailv2&iss=sbi&form=SBIIRP",
                files={"imageBin": ("face.jpg", img_bytes, "image/jpeg")})
            if r.status_code != 200:
                return [{"error": f"bing upload HTTP {r.status_code}"}]
            html = r.text
            # result cards carry titles + links in various markers
            for m in re.finditer(r'<div class="iusc"[^>]*m="([^\"]+)"', html):
                try:
                    meta = m.group(1).replace("&quot;", '"')
                    import json as _j
                    obj = _j.loads(meta)
                    purl = obj.get("purl", "")
                    t = obj.get("t", "")
                    if purl:
                        results.append({"url": purl, "title": t})
                except Exception:
                    continue
    except Exception as e:
        results.append({"error": f"bing reverse failed: {str(e)[:80]}"})
    return results[:20]


async def yandex_reverse_browser(image_path: str, attempts: int = 2) -> dict:
    """Reverse image search via a REAL browser (Playwright chromium).
    Locale-robust: Yandex serves EN or RU UI randomly, so the upload button is
    matched against both. Retries on empty results (captcha/transients).
    Returns {'entity_text': str, 'pages': [{title,url}], 'error': str|None}."""
    out = {"entity_text": "", "pages": [], "error": None}
    for attempt in range(attempts):
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
                try:
                    await page.goto("https://yandex.com/images/search?rpt=imageview",
                                    wait_until="domcontentloaded", timeout=30000)
                    # locale+timing-robust upload: poll for the input or the button
                    uploaded = False
                    for _ in range(20):   # up to ~40s
                        if await page.locator("input[type=file]").count() > 0:
                            await page.locator("input[type=file]").first.set_input_files(image_path)
                            uploaded = True
                            break
                        for label in ("Upload image", "Загрузить изображение", "Upload"):
                            try:
                                btn = page.get_by_text(label, exact=False).first
                                await btn.click(timeout=1500)
                                # input may appear immediately or after this click
                                if await page.locator("input[type=file]").count() > 0:
                                    await page.locator("input[type=file]").first.set_input_files(image_path)
                                    uploaded = True
                                break
                            except Exception:
                                continue
                        if uploaded:
                            break
                        await page.wait_for_timeout(2000)
                    if not uploaded:
                        raise RuntimeError("no upload affordance found")
                    # wait until the cbir result page loads
                    for _ in range(30):
                        if "cbir" in page.url or "rpt=imageview&url=" in page.url:
                            break
                        await page.wait_for_timeout(1000)
                    await page.wait_for_timeout(5000)
                    out["entity_text"] = (await page.inner_text("body"))[:1200]
                    # Sites tab: pages using similar images (EN/RU)
                    for tab_label in ("Sites", "Сайты"):
                        try:
                            await page.get_by_text(tab_label, exact=True).first.click(timeout=6000)
                            await page.wait_for_timeout(4000)
                            break
                        except Exception:
                            continue
                    html = await page.content()
                    for m in re.finditer(
                            r'CbirSites-Item[^>]*?href="([^"]+)"[^>]*>([^<]{4,140})<', html):
                        out["pages"].append({"url": m.group(1), "title": m.group(2).strip()})
                    if not out["pages"]:
                        for m in re.finditer(
                                r'<a[^>]+href="(https?://(?!yandex\.|ya\.ru)[^"]{10,200})"[^>]*>([^<]{6,120})<',
                                html):
                            out["pages"].append({"url": m.group(1), "title": m.group(2).strip()})
                finally:
                    await browser.close()
            if out["entity_text"] or out["pages"]:
                return out   # success
        except Exception as e:
            out["error"] = f"yandex-browser failed: {str(e)[:90]}"
    return out


async def bing_reverse_browser(image_path: str) -> dict:
    """Bing Visual Search via real browser upload.
    Returns {'entity_text': str, 'pages': [{title,url}], 'error': str|None}."""
    out = {"entity_text": "", "pages": [], "error": None}
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
            try:
                await page.goto("https://www.bing.com/images", wait_until="domcontentloaded",
                                timeout=30000)
                await page.locator("input[type=file]").first.set_input_files(image_path)
                await page.wait_for_timeout(8000)
                html = await page.content()
                # Bing often names the entity in the knowledge card
                try:
                    card = page.locator(".b_entityTitle, [data-bm] .b_focusTextLarge").first
                    out["entity_text"] = (await card.inner_text())[:200]
                except Exception:
                    pass
                for m in re.finditer(r'<div class="iusc"[^>]*?m="([^"]+)"', html):
                    try:
                        meta = m.group(1).replace("&quot;", '"')
                        import json as _j
                        obj = _j.loads(meta)
                        if obj.get("purl"):
                            out["pages"].append({"url": obj["purl"], "title": obj.get("t", "")})
                    except Exception:
                        continue
            finally:
                await browser.close()
    except Exception as e:
        out["error"] = f"bing-browser failed: {str(e)[:90]}"
    out["pages"] = out["pages"][:20]
    return out


async def reverse_image_hunt(image_path: str) -> dict:
    """Browser-driven Yandex + Bing in parallel.
    Returns {'entity_text': str (merged), 'pages': [{title,url}], 'errors': [...]}"""
    yx, bg = await asyncio.gather(yandex_reverse_browser(image_path),
                                  bing_reverse_browser(image_path))
    pages, seen = [], set()
    for r in (yx.get("pages", []) + bg.get("pages", [])):
        key = r.get("url") or r.get("title")
        if key and key not in seen:
            seen.add(key)
            pages.append(r)
    entity = (yx.get("entity_text") or "")[:600]
    if not entity:
        entity = (bg.get("entity_text") or "")[:600]
    errors = [r["error"] for r in (yx, bg) if r.get("error")]
    return {"entity_text": entity, "pages": pages[:25], "errors": errors}


# ── Search4Faces (optional, keyed) ───────────────────────────────────────────
def reverse_search_links(public_url: str = "") -> list:
    """If Sir's image has a public URL, these are one-click face hunts.
    Without a public URL, the local-upload flow is described instead."""
    if not public_url:
        return []
    return [(name, url.format(public_url=public_url)) for name, url in REVERSE_ENGINES]


async def search4faces(image_path: str, engine: str = "ttk") -> str:
    """Search4Faces: real face-match across social photo databases.
    engine: 'ttk' (TikTok), 'vk', 'ig' (Instagram). Requires API key."""
    key = os.getenv("SEARCH4FACES_API_KEY")
    if not key:
        return ""
    endpoint = {"ttk": "https://api.search4faces.com/api/v1/search/ttk",
                "vk": "https://api.search4faces.com/api/v1/search/vk",
                "ig": "https://api.search4faces.com/api/v1/search/ig"}.get(engine)
    if not endpoint:
        return ""
    import httpx
    try:
        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(
                endpoint,
                headers={"x-authorization-token": key,
                         "Content-Type": "application/json"},
                json={"image_base64": img_b64, "results_count": 12})
        if r.status_code != 200:
            return f"(search4faces HTTP {r.status_code})"
        data = r.json()
        results = data.get("results", []) or data.get("faces", []) or []
        lines = []
        for res in results[:12]:
            score = res.get("score", res.get("similarity", 0))
            src = res.get("source_url", res.get("url", ""))
            pid = res.get("owner_id", res.get("person_id", ""))
            lines.append(f"match score={score} id={pid} {src}")
        return "\n".join(lines) if lines else "(no face matches)"
    except Exception as e:
        return f"(search4faces error: {str(e)[:80]})"


def describe_hunt_plan(has_api: bool) -> str:
    """What Ada says she is doing — honest about engines available."""
    engines = ["local deepface biometric verification of harvested avatars",
               "Yandex reverse-image (strongest for faces)", "Google Lens", "Bing Visual"]
    if has_api:
        engines.insert(0, "Search4Faces face-match (VK/TikTok/Instagram)")
    return " + ".join(engines)
