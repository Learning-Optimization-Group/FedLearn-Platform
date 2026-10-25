"""Reading-copy preparation checks that never create a PDF."""
import base64
import importlib
from pathlib import Path

import pytest


def module():
    return importlib.import_module("render_integrated_report")


def test_source_references_supply_direct_links_and_keep_all_sections(tmp_path):
    source = tmp_path / "report.md"
    source.write_text("# Study\n\n## Results\n\nGain **0.12345** [1].\n\n"
                      "| Arm | AUROC |\n| --- | --- |\n| Left | 0.7 |\n\n"
                      "- Caveat *fixed heads*.\n\n## References\n\n"
                      "[1] [Specific study](https://example.org/paper)\n\n"
                      "## Appendix\n\nAfter references.\n")
    document = module().prepare(source, required_tokens=["0.12345"])
    html = module().render_html(document)
    assert '<!doctype html>' in html.lower()
    assert '<table>' in html and '<ul>' in html
    assert '<strong>0.12345</strong>' in html and '<em>fixed heads</em>' in html
    assert 'href="https://example.org/paper"' in html
    assert "Specific study" in html and "After references." in html
    assert "Wang et al." not in html and str(tmp_path) not in html


def test_markdown_reference_definitions_resolve_to_direct_links(tmp_path):
    source = tmp_path / "report.md"
    source.write_text("# Example\n\nUse [source][paper].\n\n## References\n\n"
                      "[paper]: https://example.org/paper \"Study title\"\n")
    html = module().render_html(module().prepare(source))
    assert 'href="https://example.org/paper"' in html
    assert "Study title" in html


def test_figures_are_embedded_and_never_expose_paths(tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jE1sAAAAASUVORK5CYII="))
    source = tmp_path / "report.md"
    source.write_text("# Example\n\n![Figure 1. Synthetic result](figure.png)\n")
    document = module().prepare(source)
    html = module().render_html(document)
    assert 'src="data:image/png;base64,' in html
    assert "Figure 1. Synthetic result" in html
    assert "figure.png" not in html and str(tmp_path) not in html


@pytest.mark.parametrize("text", ["# A\n\n/Users/private/report.csv", "# A\n\nfile:///tmp/report",
                                  "# A\n\n![x](/tmp/x.png)", "# A\n\n![x](../x.png)"])
def test_local_path_content_and_escaping_figures_are_rejected(tmp_path, text):
    source = tmp_path / "report.md"
    source.write_text(text)
    with pytest.raises(ValueError):
        module().prepare(source)


def test_missing_numeric_tokens_and_malformed_tables_fail_before_outputs(tmp_path):
    source = tmp_path / "report.md"
    source.write_text("# Study\n\nResult 0.12.\n")
    with pytest.raises(ValueError, match="required token"):
        module().prepare(source, required_tokens=["0.12345"])
    source.write_text("# Study\n\n| A | B |\n| --- | --- |\n| Only one |\n")
    with pytest.raises(ValueError, match="table"):
        module().prepare(source)


def test_existing_or_colliding_outputs_are_refused_without_writing(tmp_path):
    source = tmp_path / "report.md"
    source.write_text("# Study\n")
    output = tmp_path / "report.pdf"
    html = tmp_path / "report.html"
    pages = tmp_path / "pages"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        module().validate_outputs(source, output, html, pages)
    assert output.read_bytes() == b"existing"
    assert not html.exists() and not pages.exists()
    output.unlink()
    with pytest.raises(ValueError):
        module().validate_outputs(source, output, output, pages)
    with pytest.raises(ValueError):
        module().validate_outputs(source, pages / "report.pdf", html, pages)


def test_html_escapes_raw_markup_and_rejects_active_links(tmp_path):
    source = tmp_path / "report.md"
    source.write_text('# Study\n\n<script>bad()</script> [x](javascript:bad)\n')
    html = module().render_html(module().prepare(source))
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert 'href="javascript:' not in html
