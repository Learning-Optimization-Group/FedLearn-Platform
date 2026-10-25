"""Render a source-controlled Markdown manuscript into PDF and standalone HTML.

References and required result tokens come from the caller, never a historical
manuscript. --check validates preparation without creating any deliverable.
Call the required artifact-operation marker before invoking an actual render.
"""
import argparse
import base64
from datetime import date
import hashlib
from html import escape
from importlib.metadata import version
from importlib.util import find_spec
import json
import mimetypes
from pathlib import Path
import re


FONT_FILES = [
    ("IntegratedSerif", "DejaVuSerif.ttf"),
    ("IntegratedSerifBold", "DejaVuSerif-Bold.ttf"),
    ("IntegratedSerifItalic", "DejaVuSerif-Italic.ttf"),
    ("IntegratedSerifBoldItalic", "DejaVuSerif-BoldItalic.ttf"),
]
LOCAL_PATH = re.compile(r"file://|/(?:Users|home|private|tmp|var/folders)/|(?<![A-Za-z])[A-Za-z]:[\\/]")
REFERENCE = re.compile(r'^\[([^]]+)\]:\s*(https?://\S+?)(?:\s+"([^"]+)")?\s*$')
FIGURE = re.compile(r"!\[([^]]*)\]\(([^)]+)\)")
INLINE = re.compile(r"\[([^]]+)\]\(([^)]+)\)|\[([^]]+)\]\[([^]]+)\]|"
                    r"`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*|https?://[^\s<>]+")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return text.translate(str.maketrans({char: "-" for char in "‐‑‒–—−"} | {"\u00a0": " "}))


def inline(text, references=None, pdf=False):
    """Escape raw markup and permit only ordinary web hyperlinks."""
    references = references or {}
    pieces, previous = [], 0
    for match in INLINE.finditer(normalized(text)):
        pieces.append(escape(normalized(text)[previous:match.start()]))
        link_label, link_url, ref_label, ref_id, code, bold, italic = match.groups()
        if link_label is not None or ref_label is not None:
            label = link_label if link_label is not None else ref_label
            url = link_url if link_label is not None else references.get(ref_id, ("", ""))[0]
            pieces.append(f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'
                          if url.startswith(("https://", "http://")) else escape(label))
        elif code is not None:
            pieces.append(escape(code) if pdf else f"<code>{escape(code)}</code>")
        elif bold is not None:
            tag = "b" if pdf else "strong"
            pieces.append(f"<{tag}>{escape(bold)}</{tag}>")
        elif italic is not None:
            tag = "i" if pdf else "em"
            pieces.append(f"<{tag}>{escape(italic)}</{tag}>")
        else:
            url = match.group().rstrip(".,;")
            suffix = match.group()[len(url):]
            pieces.append(f'<a href="{escape(url, quote=True)}">{escape(url)}</a>{escape(suffix)}')
        previous = match.end()
    pieces.append(escape(normalized(text)[previous:]))
    return "".join(pieces)


def prepare(source, required_tokens=()):
    """Read and validate a small academic Markdown subset; produce no files."""
    source = Path(source).resolve()
    raw = source.read_bytes()
    text = normalized(raw.decode("utf-8"))
    if LOCAL_PATH.search(text):
        raise ValueError("local absolute paths are forbidden in a reading copy")
    for token in required_tokens:
        if not token or token not in text:
            raise ValueError(f"required token absent from manuscript: {token}")
    references, lines = {}, []
    for line in text.splitlines():
        match = REFERENCE.fullmatch(line.strip())
        if match:
            key, url, title = match.groups()
            if key in references:
                raise ValueError("duplicate reference definition")
            references[key] = (url, title or key)
        else:
            lines.append(line)
    blocks, figures, i = [], {}, 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        heading = re.fullmatch(r"(#{1,6})\s+(.+)", line)
        figure = FIGURE.fullmatch(line)
        bullet = re.match(r"^([-*]|\d+\.)\s+(.+)", line)
        if heading:
            blocks.append(("heading", len(heading[1]), heading[2]))
            i += 1
        elif figure:
            caption, target = figure.groups()
            path = (source.parent / target).resolve()
            if Path(target).is_absolute() or not path.is_relative_to(source.parent):
                raise ValueError("figure must be relative to and contained in the source directory")
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in ("image/png", "image/jpeg"):
                raise ValueError("figures must be PNG or JPEG for matching PDF/HTML output")
            data = path.read_bytes()
            figures[target] = dict(path=path, data=data, mime=mime, sha256=sha256(data))
            blocks.append(("figure", caption, target))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [value.strip().replace(r"\|", "|") for value in
                       re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))]
                if not all(re.fullmatch(r":?-+:?", value.replace(" ", "")) for value in row):
                    rows.append(row)
                i += 1
            if not rows or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("table rows must have matching column counts")
            blocks.append(("table", rows))
        elif bullet:
            content, marker = bullet[2], bullet[1]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                content += " " + lines[i].strip()
                i += 1
            blocks.append(("bullet", marker, content))
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(
                    r"^(?:#|\||!\[|[-*]\s|\d+\.\s)", lines[i].strip()):
                paragraph.append(lines[i].strip())
                i += 1
            blocks.append(("paragraph", " ".join(paragraph)))
    if references:
        # Definitions are source-provided references too, even when no prose list exists.
        if not any(b[0] == "heading" and b[2].lower() == "references" for b in blocks):
            blocks.append(("heading", 2, "References"))
        blocks.extend(("paragraph", f"[{key}] [{title}]({url})")
                      for key, (url, title) in references.items())
    if not blocks:
        raise ValueError("manuscript must contain visible content")
    title = next((b[2] for b in blocks if b[0] == "heading" and b[1] == 1), "Research report")
    return dict(blocks=blocks, references=references, figures=figures, title=title,
                source_sha256=sha256(raw), required_tokens=list(required_tokens))


def validate_outputs(source, output, html, pages):
    source, output, html, pages = map(lambda p: Path(p).resolve(), (source, output, html, pages))
    paths = (output, html, pages)
    if len(set(paths)) != 3 or source in paths:
        raise ValueError("source and outputs must be distinct")
    if any(a.is_relative_to(b) for a in paths for b in paths if a != b):
        raise ValueError("output files and page directory must not overlap")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError("use fresh PDF, HTML, and page outputs")


def check_environment(font_dir):
    font_dir = Path(font_dir)
    missing = [package for package in ("reportlab", "pypdf", "pypdfium2") if find_spec(package) is None]
    missing += [filename for _, filename in FONT_FILES if not (font_dir / filename).is_file()]
    if missing:
        raise RuntimeError("missing renderer dependencies or fonts: " + ", ".join(missing))
    return {package: version(package) for package in ("reportlab", "pypdf", "pypdfium2")}


def render_html(document, font_dir=None):
    fonts = ""
    if font_dir is not None:
        for (_, filename), (weight, style) in zip(FONT_FILES, [(400, "normal"), (700, "normal"),
                                                               (400, "italic"), (700, "italic")]):
            encoded = base64.b64encode((Path(font_dir) / filename).read_bytes()).decode("ascii")
            fonts += ("@font-face{font-family:ReportSerif;src:url(data:font/ttf;base64," + encoded
                      + f") format('truetype');font-weight:{weight};font-style:{style};}}")
    body, list_kind = [], None
    fmt = lambda text: inline(text, document["references"])
    for block in document["blocks"]:
        kind = block[0]
        next_list = ("ol" if block[1].endswith(".") else "ul") if kind == "bullet" else None
        if list_kind != next_list:
            if list_kind:
                body.append(f"</{list_kind}>")
            if next_list:
                start = f' start="{int(block[1][:-1])}"' if next_list == "ol" else ""
                body.append(f"<{next_list}{start}>")
            list_kind = next_list
        if kind == "heading":
            body.append(f"<h{block[1]}>{fmt(block[2])}</h{block[1]}>")
        elif kind == "paragraph":
            body.append(f"<p>{fmt(block[1])}</p>")
        elif kind == "bullet":
            body.append(f"<li>{fmt(block[2])}</li>")
        elif kind == "table":
            body.append("<table>")
            for index, row in enumerate(block[1]):
                tag = "th" if index == 0 else "td"
                body.append("<tr>" + "".join(f"<{tag}>{fmt(v)}</{tag}>" for v in row) + "</tr>")
            body.append("</table>")
        elif kind == "figure":
            image = document["figures"][block[2]]
            encoded = base64.b64encode(image["data"]).decode("ascii")
            body.append(f'<figure><img src="data:{image["mime"]};base64,{encoded}" '
                        f'alt="{escape(block[1], quote=True)}"><figcaption>{fmt(block[1])}</figcaption></figure>')
    if list_kind:
        body.append(f"</{list_kind}>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta name="author" content="Anurag Lnu">'
            f'<title>{escape(document["title"])}</title><style>{fonts}'
            "body{font-family:ReportSerif,Georgia,serif;font-size:17px;line-height:1.6;color:#18232b;"
            "max-width:860px;margin:3rem auto;padding:0 1.4rem}h1{font-size:2rem;line-height:1.25}"
            "h2{font-size:1.35rem;margin-top:2rem}h3,h4{font-size:1.12rem;margin-top:1.5rem}"
            "a{color:#145670;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;font-size:.88em;"
            "margin:1.3rem 0}th,td{text-align:left;vertical-align:top;padding:.5rem;border-bottom:1px solid #ccd3d7}"
            "th{background:#edf2f3;border-top:1px solid #33434b}figure{margin:1.5rem 0}img{max-width:100%;height:auto}"
            "figcaption{font-size:.88em;color:#43515a}code{font-family:monospace;font-size:.9em}"
            "@media print{body{max-width:none;margin:0;font-size:10.3pt}h1,h2,h3{break-after:avoid}"
            "figure,tr{break-inside:avoid}a{color:inherit}}</style></head><body><main>"
            + "\n".join(body) + "</main></body></html>\n")


def render_pdf(document, output, font_dir, report_date):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    for name, filename in FONT_FILES:
        pdfmetrics.registerFont(TTFont(name, str(Path(font_dir) / filename)))
    pdfmetrics.registerFontFamily("IntegratedSerif", normal="IntegratedSerif", bold="IntegratedSerifBold",
                                  italic="IntegratedSerifItalic", boldItalic="IntegratedSerifBoldItalic")
    width = letter[0] - 2 * 47
    body = ParagraphStyle("body", fontName="IntegratedSerif", fontSize=10.3, leading=13.5,
                          spaceAfter=6, allowWidows=0, allowOrphans=0)
    small = ParagraphStyle("small", parent=body, fontSize=8.4, leading=11)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.1, leading=10.5, spaceAfter=0)
    story = []
    fmt = lambda text: inline(text, document["references"], pdf=True)
    for block in document["blocks"]:
        if block[0] == "heading":
            level = block[1]
            if level == 2 and block[2].strip().casefold() == "references" and story:
                story.append(PageBreak())
            style = ParagraphStyle(f"h{level}", parent=body, fontName="IntegratedSerifBold",
                                   fontSize=17 if level == 1 else 11.5, leading=21 if level == 1 else 15,
                                   spaceBefore=5 if level == 1 else 11, spaceAfter=7, keepWithNext=True,
                                   alignment=TA_CENTER if level == 1 else 0)
            story.append(Paragraph(fmt(block[2]), style))
        elif block[0] == "paragraph":
            story.append(Paragraph(fmt(block[1]), body))
        elif block[0] == "bullet":
            style = ParagraphStyle("bullet", parent=body, leftIndent=15, firstLineIndent=0)
            story.append(Paragraph(fmt(block[2]), style, bulletText=block[1]))
        elif block[0] == "figure":
            image = Image(str(document["figures"][block[2]]["path"]))
            scale = min(width / image.imageWidth, 560 / image.imageHeight)
            image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
            story.append(KeepTogether([Spacer(1, 5), image, Spacer(1, 4), Paragraph(fmt(block[1]), small)]))
        elif block[0] == "table":
            rows = block[1]
            weights = [max(8, min(35, max(len(row[index]) for row in rows))) for index in range(len(rows[0]))]
            table = Table([[Paragraph(fmt(value), cell) for value in row] for row in rows],
                          colWidths=[width * weight / sum(weights) for weight in weights], repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f3")),
                ("LINEABOVE", (0, 0), (-1, 0), .6, colors.HexColor("#33434b")),
                ("LINEBELOW", (0, -1), (-1, -1), .6, colors.HexColor("#33434b")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            table_block = [Spacer(1, 4), table, Spacer(1, 8)]
            if len(rows) <= 8:
                if (story and isinstance(story[-1], Paragraph)
                        and getattr(story[-1].style, "keepWithNext", False)):
                    table_block.insert(0, story.pop())
                story.append(KeepTogether(table_block))
            else:
                story.extend(table_block)

    def page(canvas, doc):
        canvas.setAuthor("Anurag Lnu")
        canvas.setTitle(document["title"])
        canvas.setCreator("")
        canvas._doc.info.producer = ""
        canvas.saveState()
        canvas.setFont("IntegratedSerif", 7.6)
        canvas.setFillColor(colors.HexColor("#53616a"))
        canvas.drawString(47, 25, f"MSCS Capstone | Reading copy | {report_date}")
        canvas.drawRightString(letter[0] - 47, 25, str(doc.page))
        canvas.restoreState()

    destination = output if hasattr(output, "write") else str(output)
    doc = SimpleDocTemplate(destination, invariant=1, pagesize=letter, leftMargin=47, rightMargin=47,
                            topMargin=45, bottomMargin=47)
    doc.build(story, onFirstPage=page, onLaterPages=page)


def main(source, output, html, pages, font_dir, report_date, required_tokens=(), check=False):
    validate_outputs(source, output, html, pages)
    dependencies = check_environment(font_dir)
    report_date = date.fromisoformat(report_date).strftime("%d %B %Y")
    document = prepare(source, required_tokens)
    reading_copy = render_html(document, font_dir)
    if check:
        return dict(status="preflight_only", dependencies=dependencies, blocks=len(document["blocks"]),
                    figures=len(document["figures"]), source_sha256=document["source_sha256"])
    from pypdf import PdfReader
    import pypdfium2 as pdfium

    output, html, pages = map(Path, (output, html, pages))
    for path in (output.parent, html.parent):
        path.mkdir(parents=True, exist_ok=True)
    pages.mkdir(parents=True, exist_ok=False)
    # Exclusive creation protects existing files, including a race after preflight.
    with output.open("xb") as stream:
        render_pdf(document, stream, font_dir, report_date)
    with html.open("x", encoding="utf-8") as stream:
        stream.write(reading_copy)
    reader = PdfReader(output)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for token in required_tokens:
        if token not in text:
            raise ValueError(f"required token absent from extracted PDF: {token}")
    if any(char in text for char in ("\ufffd", "\u25a0")) or LOCAL_PATH.search(text):
        raise ValueError("unreadable glyph or local path in extracted PDF")
    metadata = reader.metadata
    if metadata.author != "Anurag Lnu" or metadata.creator or metadata.producer:
        raise ValueError("unexpected PDF author, creator, or producer metadata")
    rendered = {}
    with pdfium.PdfDocument(str(output)) as pdf:
        for index in range(len(pdf)):
            target = pages / f"page-{index + 1:02}.png"
            page = pdf[index]
            bitmap = page.render(scale=1.5)
            bitmap.to_pil().save(target)
            bitmap.close()
            page.close()
            rendered[target.name] = sha256(target.read_bytes())
    receipt = dict(source_sha256=document["source_sha256"], renderer_sha256=sha256(Path(__file__).read_bytes()),
                   pdf_sha256=sha256(output.read_bytes()), html_sha256=sha256(html.read_bytes()),
                   font_sha256={filename: sha256((Path(font_dir) / filename).read_bytes()) for _, filename in FONT_FILES},
                   figure_sha256={target: item["sha256"] for target, item in document["figures"].items()},
                   dependencies=dependencies, report_date=report_date, pages=len(reader.pages),
                   text_characters=len(text), required_tokens=list(required_tokens), rendered_pages=rendered,
                   scope="Source manuscript reading copies; page PNGs require visual review before acceptance.")
    (pages / "render-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "html", "pages", "font-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report-date", required=True, help="Explicit date, YYYY-MM-DD")
    parser.add_argument("--required-token", action="append", default=[])
    parser.add_argument("--check", action="store_true", help="Validate without creating PDF, HTML, or PNG outputs")
    args = parser.parse_args()
    print(json.dumps(main(args.source, args.output, args.html, args.pages, args.font_dir,
                          args.report_date, args.required_token, args.check), sort_keys=True))
