"""Minimal dependency-free XLSX export for the displayed interaction network."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping, Sequence
from xml.sax.saxutils import escape

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_HEADERS = (
    ("Subject", "subject"),
    ("Subject normalized ID", "subject_normalized_id"),
    ("Subject type", "subject_type"),
    ("Predicate", "predicate"),
    ("Object", "object"),
    ("Object normalized ID", "object_normalized_id"),
    ("Object type", "object_type"),
    ("Direction", "direction"),
)


def _column_name(index: int) -> str:
    name = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _inline_cell(reference: str, value: object) -> str:
    text = "" if value is None else str(value)
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return (
        f'<c r="{reference}" t="inlineStr"><is><t{preserve}>'
        f"{escape(text)}"
        "</t></is></c>"
    )


def _worksheet_xml(rows: Sequence[Mapping[str, object]]) -> str:
    worksheet_rows: list[str] = []
    header_cells = "".join(
        _inline_cell(f"{_column_name(column)}1", header)
        for column, (header, _key) in enumerate(_HEADERS, start=1)
    )
    worksheet_rows.append(f'<row r="1">{header_cells}</row>')

    for row_number, row in enumerate(rows, start=2):
        cells = "".join(
            _inline_cell(
                f"{_column_name(column)}{row_number}",
                row.get(key, ""),
            )
            for column, (_header, key) in enumerate(_HEADERS, start=1)
        )
        worksheet_rows.append(f'<row r="{row_number}">{cells}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(worksheet_rows)}</sheetData>"
        "</worksheet>"
    )


def build_displayed_network_xlsx(
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    """Return a plain one-sheet XLSX with exactly eight text columns."""

    if not rows:
        raise ValueError("At least one displayed relation is required.")

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>'''
    package_relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Displayed network" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
    workbook_relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_relationships)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        archive.writestr("xl/worksheets/sheet1.xml", _worksheet_xml(rows))
    return output.getvalue()


__all__ = ["_XLSX_MIME", "build_displayed_network_xlsx"]
