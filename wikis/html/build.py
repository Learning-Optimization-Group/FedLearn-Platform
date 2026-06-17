#!/usr/bin/env python3
"""
Generate a self-contained, Ember-themed HTML rendering of the FedLearn wikis.

Walks every Markdown file under ``wikis/`` (excluding this ``html/`` output
directory), renders each to a styled HTML page, and mirrors the directory tree
under ``wikis/html/``. Output is fully offline: a shared ``styles.css``, a local
``app.js`` (theme toggle + on-page TOC + mobile nav), and locally-bundled brand
fonts (fetched once at build time, with a system-font fallback if offline).

Run:  python3 wikis/html/build.py
Deps: ``markdown`` + ``pygments`` (already used by CI tooling).
"""
from __future__ import annotations

import os
import re
import shutil
import urllib.request
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

HTML_DIR = Path(__file__).resolve().parent          # wikis/html
WIKI_DIR = HTML_DIR.parent                           # wikis/

# ---------------------------------------------------------------------------
# Ember design tokens (mirrored from frontend/src/styles/tokens.css)
# ---------------------------------------------------------------------------
LIGHT = {
    "canvas": "#FBF9F6", "surface-1": "#FFFFFF", "surface-2": "#F4F1EC",
    "surface-3": "#ECE7DF", "code-well": "#F6F3EE", "hairline": "#E7E1D8",
    "line": "#D8D1C5", "fg": "#1A1714", "fg-muted": "#6B6358",
    "fg-subtle": "#938A7D", "accent": "#C56A1E", "accent-hover": "#A9591A",
    "accent-fg": "#FFFFFF", "success": "#1F9D57", "warning": "#B07D0A",
    "danger": "#CE3F38",
}
DARK = {
    "canvas": "#000000", "surface-1": "#0B0A09", "surface-2": "#141210",
    "surface-3": "#1E1B17", "code-well": "#050504", "hairline": "#2A2520",
    "line": "#352F27", "fg": "#F5F1EA", "fg-muted": "#A8A096",
    "fg-subtle": "#6E665C", "accent": "#F7A35C", "accent-hover": "#FFB877",
    "accent-fg": "#1C0F03", "success": "#5FD39B", "warning": "#EBC152",
    "danger": "#FF6B5E",
}

FONT_SANS = "'Hanken Grotesk', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
FONT_DISPLAY = "'Bricolage Grotesque', 'Hanken Grotesk', ui-sans-serif, system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace"

# Section dir -> display title, in sidebar order. "." holds the top-level pages.
GROUPS = [
    (".", "Overview"),
    ("backend", "Backend"),
    ("frontend", "Frontend"),
    ("framework", "Framework"),
    ("desktop", "Desktop"),
    ("mobile", "Mobile"),
    ("client-docker", "Client (Docker)"),
]

# ---------------------------------------------------------------------------
# Page discovery
# ---------------------------------------------------------------------------
def first_h1(text: str, fallback: str) -> str:
    for line in text.splitlines():
        m = re.match(r"#\s+(.*\S)\s*$", line)
        if m:
            return m.group(1).strip()
    return fallback


def out_rel_for(src_rel: str) -> str:
    """Map a wiki-relative .md path to its html output path."""
    if src_rel == "README.md":
        return "index.html"
    return src_rel[:-3] + ".html"


def nav_label(page) -> str:
    if page["out"] == "index.html":
        return "Master Wiki"
    if page["out"] == "VERSIONS.html":
        return "Component Versions"
    # Strip a leading "01 - " / "01_" numbering for a tidy sidebar label.
    return re.sub(r"^\d+\s*[-_.]\s*", "", page["title"])


def discover():
    pages = []  # ordered
    for d, group_title in GROUPS:
        base = WIKI_DIR if d == "." else WIKI_DIR / d
        if not base.exists():
            continue
        if d == ".":
            mds = [p for p in (base / "README.md", base / "VERSIONS.md") if p.exists()]
        else:
            mds = sorted(
                base.glob("*.md"),
                key=lambda p: (p.name.lower() != "readme.md", p.name.lower()),
            )
        for md in mds:
            src_rel = md.relative_to(WIKI_DIR).as_posix()
            text = md.read_text(encoding="utf-8")
            pages.append({
                "src": md,
                "src_rel": src_rel,
                "out": out_rel_for(src_rel),
                "title": first_h1(text, md.stem),
                "group": group_title,
                "text": text,
            })
    return pages


# ---------------------------------------------------------------------------
# Link rewriting: .md -> .html, root README.md -> index.html, html/ self-links.
# Only touches real <a href> / <img src> tags (code samples are HTML-escaped,
# so their literal `href="..."` text never matches `<a ...`).
# ---------------------------------------------------------------------------
def make_link_rewriter(linkmap: dict[str, str], src_rel: str):
    cur_dir = os.path.dirname(src_rel)

    def remap(url: str) -> str:
        if not url or url.startswith(("#", "http://", "https://", "mailto:", "//", "/", "data:")):
            return url
        base, _, anchor = url.partition("#")
        anchor = ("#" + anchor) if anchor else ""
        if not base:
            return url
        resolved = os.path.normpath(os.path.join(cur_dir, base)).replace("\\", "/")
        target = None
        if resolved in linkmap:                       # a rendered .md page
            target = linkmap[resolved]
        elif resolved.startswith("html/"):            # link pointing into html/ itself
            target = resolved[len("html/"):]
        if target is None:
            return url                                # external / out-of-wiki: leave as-is
        rel = os.path.relpath(target, cur_dir or ".").replace("\\", "/")
        return rel + anchor

    def repl_a(m):
        return m.group(1) + remap(m.group(2)) + m.group(3)

    def rewrite(html_str: str) -> str:
        html_str = re.sub(r'(<a\s+[^>]*?href=")([^"]+)(")', repl_a, html_str)
        html_str = re.sub(r'(<img\s+[^>]*?src=")([^"]+)(")', repl_a, html_str)
        return html_str

    return rewrite


def tag_callouts(html_str: str) -> str:
    """Style warning blockquotes (those starting with ⚠️) as callouts."""
    html_str = html_str.replace(
        "<blockquote>\n<p>⚠️", '<blockquote class="callout warn"><p>⚠️'
    )
    return html_str


# ---------------------------------------------------------------------------
# Brand fonts: fetch latin subsets to html/fonts/ (graceful fallback if offline)
# ---------------------------------------------------------------------------
GFONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Bricolage+Grotesque:opsz,wght@12..96,200..800"
    "&family=Hanken+Grotesk:wght@400;500;600;700;800"
    "&family=JetBrains+Mono:wght@400;500;700"
    "&display=swap"
)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) "
      "Gecko/20100101 Firefox/128.0")


def bundle_fonts(fonts_dir: Path) -> bool:
    fonts_dir.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(GFONTS_URL, headers={"User-Agent": UA})
        css = urllib.request.urlopen(req, timeout=20).read().decode("utf-8")
    except Exception as e:  # offline / blocked — fall back to system fonts
        (fonts_dir / "fonts.css").write_text(
            f"/* Font bundling skipped ({e}); using system fallback stacks. */\n",
            encoding="utf-8",
        )
        return False

    blocks = re.findall(
        r"/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*\{[^}]*\})", css, re.S
    )
    kept, seen = [], set()
    for subset, block in blocks:
        if subset not in ("latin", "latin-ext"):
            continue
        url_m = re.search(r"url\((https://[^)]+\.woff2)\)", block)
        if not url_m:
            continue
        fam = re.search(r"font-family:\s*'([^']+)'", block)
        wght = re.search(r"font-weight:\s*([\d ]+)", block)
        fam_s = (fam.group(1) if fam else "font").replace(" ", "")
        wght_s = (wght.group(1).strip().replace(" ", "-") if wght else "x")
        fname = f"{fam_s}-{wght_s}-{subset}.woff2"
        if fname not in seen:
            seen.add(fname)
            try:
                data = urllib.request.urlopen(
                    urllib.request.Request(url_m.group(1), headers={"User-Agent": UA}),
                    timeout=20,
                ).read()
                (fonts_dir / fname).write_bytes(data)
            except Exception:
                continue
        kept.append(
            f"/* {subset} */\n" + block.replace(url_m.group(1), f"./{fname}")
        )
    (fonts_dir / "fonts.css").write_text("\n".join(kept) + "\n", encoding="utf-8")
    return bool(kept)


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
def vars_block(selector: str, tokens: dict) -> str:
    lines = "".join(f"  --{k}: {v};\n" for k, v in tokens.items())
    return f"{selector} {{\n{lines}}}\n"


def build_styles() -> str:
    pyg_light = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    pyg_dark = HtmlFormatter(style="dracula").get_style_defs(".dark .codehilite")
    base = """
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: var(--font-sans);
  background: var(--canvas);
  color: var(--fg);
  -webkit-font-smoothing: antialiased;
  line-height: 1.65;
  font-feature-settings: 'calt' 1;
}
a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-hover); text-decoration: underline; }

.layout { display: grid; grid-template-columns: 288px minmax(0, 1fr); min-height: 100vh; }

/* Sidebar */
.sidebar {
  position: sticky; top: 0; align-self: start; height: 100vh; overflow-y: auto;
  background: var(--surface-2);
  border-right: 1px solid var(--hairline);
  padding: 28px 20px 48px;
}
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.brand .mark {
  width: 30px; height: 30px; border-radius: 9px; flex: none;
  background: radial-gradient(120% 120% at 30% 20%, #F7A35C 0%, var(--accent) 55%, #8E3F12 100%);
  box-shadow: 0 2px 10px rgba(197,106,30,.35);
}
.brand .name { font-family: var(--font-display); font-weight: 700; font-size: 18px; letter-spacing: -.01em; }
.brand .sub { font-size: 11px; color: var(--fg-subtle); letter-spacing: .14em; text-transform: uppercase; }
.nav-group { margin-top: 18px; }
.nav-group > summary {
  list-style: none; cursor: pointer; user-select: none;
  font-size: 11.5px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: var(--fg-subtle); padding: 4px 8px; border-radius: 6px;
}
.nav-group > summary::-webkit-details-marker { display: none; }
.nav-group > summary:hover { color: var(--fg-muted); }
.nav-group ul { list-style: none; margin: 6px 0 0; padding: 0; }
.nav-group li a {
  display: block; padding: 6px 10px; border-radius: 8px; font-size: 14px;
  color: var(--fg-muted); border-left: 2px solid transparent; line-height: 1.35;
}
.nav-group li a:hover { background: var(--surface-3); color: var(--fg); text-decoration: none; }
.nav-group li a.active {
  color: var(--accent); background: var(--surface-1);
  border-left-color: var(--accent); font-weight: 600;
}

/* Main column */
.main { min-width: 0; }
.topbar {
  position: sticky; top: 0; z-index: 5;
  display: flex; align-items: center; gap: 14px;
  padding: 14px 32px; background: color-mix(in srgb, var(--canvas) 86%, transparent);
  backdrop-filter: blur(10px); border-bottom: 1px solid var(--hairline);
}
.crumb { font-size: 13px; color: var(--fg-subtle); }
.crumb b { color: var(--fg-muted); font-weight: 600; }
.spacer { flex: 1; }
.icon-btn {
  display: inline-flex; align-items: center; justify-content: center;
  width: 36px; height: 36px; border-radius: 9px; cursor: pointer;
  background: var(--surface-1); border: 1px solid var(--line); color: var(--fg-muted);
}
.icon-btn:hover { color: var(--accent); border-color: var(--accent); }
#menu-btn { display: none; }

.content-wrap { display: grid; grid-template-columns: minmax(0,1fr) 220px; gap: 40px; max-width: 1180px; margin: 0 auto; padding: 12px 32px 96px; }
.article { min-width: 0; }
.toc { position: sticky; top: 78px; align-self: start; font-size: 13px; max-height: calc(100vh - 110px); overflow-y: auto; }
.toc .toc-title { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--fg-subtle); margin-bottom: 10px; }
.toc a { display: block; padding: 3px 0 3px 12px; color: var(--fg-muted); border-left: 2px solid var(--hairline); }
.toc a:hover { color: var(--accent); text-decoration: none; }
.toc a.lvl-3 { padding-left: 26px; font-size: 12.5px; }
.toc a.active { color: var(--accent); border-left-color: var(--accent); }

/* Typography */
.article h1, .article h2, .article h3, .article h4 { font-family: var(--font-display); letter-spacing: -.015em; line-height: 1.2; scroll-margin-top: 80px; }
.article h1 { font-size: 2.1rem; margin: .2em 0 .6em; }
.article h2 { font-size: 1.5rem; margin: 2em 0 .7em; padding-bottom: .3em; border-bottom: 1px solid var(--hairline); }
.article h3 { font-size: 1.18rem; margin: 1.6em 0 .5em; }
.article h4 { font-size: 1rem; margin: 1.3em 0 .4em; color: var(--fg-muted); }
.article p, .article li { color: var(--fg); }
.headerlink { margin-left: .4em; opacity: 0; color: var(--fg-subtle); font-weight: 400; }
.article h1:hover .headerlink, .article h2:hover .headerlink, .article h3:hover .headerlink, .article h4:hover .headerlink { opacity: 1; }

/* Inline + block code */
code { font-family: var(--font-mono); font-size: .88em; }
.article :not(pre) > code {
  background: var(--surface-3); padding: .12em .4em; border-radius: 5px;
  border: 1px solid var(--hairline); color: var(--accent-hover);
}
.codehilite, .article pre {
  background: var(--code-well); border: 1px solid var(--hairline);
  border-radius: var(--radius-md, 9px); padding: 16px 18px; overflow-x: auto;
  margin: 1.2em 0; font-size: 13.5px; line-height: 1.55;
}
.codehilite pre { background: none; border: 0; padding: 0; margin: 0; }
.dark .article :not(pre) > code { color: var(--accent); }

/* Tables */
.article table { border-collapse: collapse; width: 100%; margin: 1.4em 0; font-size: 14px; display: block; overflow-x: auto; }
.article th, .article td { border: 1px solid var(--hairline); padding: 8px 12px; text-align: left; vertical-align: top; }
.article thead th { background: var(--surface-2); font-weight: 600; }
.article tbody tr:nth-child(even) { background: color-mix(in srgb, var(--surface-2) 55%, transparent); }

/* Blockquotes / callouts */
.article blockquote {
  margin: 1.3em 0; padding: 10px 18px; border-left: 3px solid var(--line);
  background: var(--surface-2); border-radius: 0 9px 9px 0; color: var(--fg-muted);
}
.article blockquote p { margin: .4em 0; }
.article blockquote.callout.warn {
  border-left-color: var(--warning);
  background: color-mix(in srgb, var(--warning) 12%, var(--surface-1));
  color: var(--fg);
}
.article hr { border: 0; border-top: 1px solid var(--hairline); margin: 2.4em 0; }
.article img { max-width: 100%; border-radius: 9px; }

/* Responsive */
@media (max-width: 1080px) { .content-wrap { grid-template-columns: minmax(0,1fr); } .toc { display: none; } }
@media (max-width: 880px) {
  .layout { grid-template-columns: 1fr; }
  .sidebar {
    position: fixed; z-index: 20; width: 300px; left: 0; top: 0;
    transform: translateX(-100%); transition: transform .22s ease;
    box-shadow: 0 0 40px rgba(0,0,0,.25);
  }
  body.nav-open .sidebar { transform: translateX(0); }
  #menu-btn { display: inline-flex; }
  body.nav-open::after { content:''; position: fixed; inset: 0; z-index: 15; background: rgba(0,0,0,.4); }
}
"""
    return (
        vars_block(":root", {**LIGHT})
        + f":root {{ --font-sans: {FONT_SANS}; --font-display: {FONT_DISPLAY}; --font-mono: {FONT_MONO}; --radius-md: 9px; }}\n"
        + vars_block("html.dark", {**DARK})
        + base
        + "\n/* pygments: light */\n" + pyg_light
        + "\n/* pygments: dark */\n" + pyg_dark
    )


APP_JS = """
(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('wiki-theme'); } catch (e) {}
  var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (saved === 'dark' || (saved === null && prefersDark)) root.classList.add('dark');

  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('theme-btn');
    if (btn) btn.addEventListener('click', function () {
      root.classList.toggle('dark');
      try { localStorage.setItem('wiki-theme', root.classList.contains('dark') ? 'dark' : 'light'); } catch (e) {}
    });
    var menu = document.getElementById('menu-btn');
    if (menu) menu.addEventListener('click', function () { document.body.classList.toggle('nav-open'); });
    document.addEventListener('click', function (e) {
      if (document.body.classList.contains('nav-open') &&
          !e.target.closest('.sidebar') && e.target.id !== 'menu-btn' &&
          !e.target.closest('#menu-btn')) {
        document.body.classList.remove('nav-open');
      }
    });

    // Build the on-page TOC from h2/h3 in the article.
    var article = document.querySelector('.article');
    var toc = document.querySelector('.toc');
    if (!article || !toc) return;
    var heads = article.querySelectorAll('h2[id], h3[id]');
    if (heads.length < 2) { toc.remove(); return; }
    var html = '<div class="toc-title">On this page</div>';
    heads.forEach(function (h) {
      var lvl = h.tagName === 'H3' ? ' lvl-3' : '';
      var text = h.textContent.replace('\\u00b6', '').trim();
      html += '<a class="toc-link' + lvl + '" href="#' + h.id + '">' + text + '</a>';
    });
    toc.innerHTML = html;

    var links = toc.querySelectorAll('a');
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          links.forEach(function (a) { a.classList.remove('active'); });
          if (byId[en.target.id]) byId[en.target.id].classList.add('active');
        }
      });
    }, { rootMargin: '-72px 0px -70% 0px', threshold: 0 });
    heads.forEach(function (h) { obs.observe(h); });
  });
})();
"""


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
def rel(target: str, from_out: str) -> str:
    d = os.path.dirname(from_out)
    return os.path.relpath(target, d or ".").replace("\\", "/")


def build_nav(pages, cur_out: str) -> str:
    by_group = {}
    for p in pages:
        by_group.setdefault(p["group"], []).append(p)
    out = []
    for _, gtitle in GROUPS:
        gpages = by_group.get(gtitle)
        if not gpages:
            continue
        is_open = any(p["out"] == cur_out for p in gpages) or gtitle in ("Overview",)
        out.append(f'<details class="nav-group"{" open" if is_open else ""}>')
        out.append(f"<summary>{gtitle}</summary><ul>")
        for p in gpages:
            cls = " class=\"active\"" if p["out"] == cur_out else ""
            href = rel(p["out"], cur_out)
            out.append(f'<li><a{cls} href="{href}">{nav_label(p)}</a></li>')
        out.append("</ul></details>")
    return "\n".join(out)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · FedLearn Wiki</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%23C56A1E'/%3E%3Cpath d='M16 5c1.5 4-2.5 5.5-2.5 9a4.2 4.2 0 0 0 2.5 4c-.6-2 .8-3 1.6-4.2 1.9 1.4 2.9 3 2.9 5.2A6 6 0 1 1 9.8 20c0-4.6 4.2-6.7 6.2-15z' fill='%23FBF9F6'/%3E%3C/svg%3E">
<link rel="stylesheet" href="__STYLES__">
<link rel="stylesheet" href="__FONTS__">
<script src="__APP__" defer></script>
</head>
<body>
<div class="layout">
<aside class="sidebar">
  <a class="brand" href="__INDEX__" style="text-decoration:none;color:inherit">
    <span class="mark"></span>
    <span><span class="name">FedLearn</span><br><span class="sub">Platform Wiki</span></span>
  </a>
  <nav>__NAV__</nav>
</aside>
<div class="main">
  <header class="topbar">
    <button id="menu-btn" class="icon-btn" aria-label="Toggle navigation">&#9776;</button>
    <div class="crumb"><b>__GROUP__</b> &nbsp;·&nbsp; __TITLE__</div>
    <span class="spacer"></span>
    <button id="theme-btn" class="icon-btn" aria-label="Toggle theme" title="Toggle light / dark">&#9681;</button>
  </header>
  <div class="content-wrap">
    <article class="article">__BODY__</article>
    <nav class="toc"></nav>
  </div>
</div>
</div>
</body>
</html>
"""


def main():
    pages = discover()
    linkmap = {p["src_rel"]: p["out"] for p in pages}

    # Fresh output (preserve build.py itself).
    for child in HTML_DIR.iterdir():
        if child.name == "build.py":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    (HTML_DIR / "styles.css").write_text(build_styles(), encoding="utf-8")
    (HTML_DIR / "app.js").write_text(APP_JS, encoding="utf-8")
    ok = bundle_fonts(HTML_DIR / "fonts")
    print(f"fonts: {'bundled latin subsets' if ok else 'fallback (system fonts)'}")

    if (WIKI_DIR / "assets").exists():
        shutil.copytree(WIKI_DIR / "assets", HTML_DIR / "assets")

    md = markdown.Markdown(
        extensions=["extra", "admonition", "sane_lists", "toc", "codehilite"],
        extension_configs={
            "toc": {"permalink": True},
            "codehilite": {"guess_lang": False, "css_class": "codehilite"},
        },
    )

    for p in pages:
        md.reset()
        body = md.convert(p["text"])
        body = tag_callouts(body)
        body = make_link_rewriter(linkmap, p["src_rel"])(body)
        page_html = (
            PAGE.replace("__TITLE__", p["title"])
            .replace("__GROUP__", p["group"])
            .replace("__STYLES__", rel("styles.css", p["out"]))
            .replace("__FONTS__", rel("fonts/fonts.css", p["out"]))
            .replace("__APP__", rel("app.js", p["out"]))
            .replace("__INDEX__", rel("index.html", p["out"]))
            .replace("__NAV__", build_nav(pages, p["out"]))
            .replace("__BODY__", body)
        )
        dest = HTML_DIR / p["out"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page_html, encoding="utf-8")

    print(f"rendered {len(pages)} pages -> {HTML_DIR}")


if __name__ == "__main__":
    main()
