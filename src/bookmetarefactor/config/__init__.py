from pathlib import Path


DEFAULT_NUM_FIRST_PAGES = 4
DEFAULT_NUM_LAST_PAGES = 2

MAX_LONG_EDGE_IMG = 1600
MAX_REQUESTS_PER_BATCH = 50_000
MAX_BATCH_BYTES = 128 * 1024 * 1024

TTRPG_OCR_LLM_PROMPT = """
You are an OCR assistant reading high-resolution images of 
tabletop RPG (TTRPG) books. Pages may contain ornate layouts, 
multi-column text, sidebars, tables, stat blocks, headers and 
footers, and stylized fonts. Carefully transcribe all legible 
text and return the final transcript in Markdown while preserving 
important structure:

Read the entire page top-to-bottom, left-to-right.
Capture headers, footers, captions, tables, sidebars, callouts, 
and stat blocks. Use Markdown headings (#, ##, etc.) and horizontal 
rules (---) to mark major sections or layout cues 
(e.g., “### Sidebar: Rules Summary” or “Table: Weapon Traits”).
For multi-column pages, transcribe column by column in reading order 
and insert <!-- COLUMN BREAK --> comments between columns.

Preserve bullet lists, numbered steps, and tables using Markdown syntax 
(* or - for bullets, 1. for numbered lists, | col | col | tables).
Transcribe game mechanics verbatim (stat lines, dice expressions, 
modifiers, ability descriptions, footnotes) using inline code `1d6+2` 
where relevant.

Mark any illegible text with [ILLEGIBLE].

Do not add commentary or interpretation—only capture what is visible.

Return the final Markdown document so downstream tools can understand 
both the text content and layout structure.
"""

MAX_REQUESTS_PER_BATCH = 50_000
MAX_BATCH_BYTES = 100 * 1024 * 1024

SECRETS_PATH = Path("")
