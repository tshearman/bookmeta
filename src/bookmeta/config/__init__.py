from pathlib import Path

DEFAULT_NUM_FIRST_PAGES = 4
DEFAULT_NUM_LAST_PAGES = 2

MAX_LONG_EDGE_IMG = 1200
MAX_REQUESTS_PER_BATCH = 50_000
MAX_BATCH_BYTES = 128 * 1024 * 1024

TTRPG_BOOKINFO_PROMPT = """
You are analyzing pages from a tabletop role-playing game (TTRPG) product.
The images are the front cover and back cover and there are imperfect OCR 
transcripts of a few pages of the book (front cover, back cover, and early interior pages).
Use only visible/provided text; never hallucinate.

Priorities:
• Extract the book's title, subtitle, primary author/designer, and publisher
• Provide confidence values between 0 and 1 for title_confidence and author_confidence.
• Identify if the product is a core rulebook, adventure/module, supplement, setting,
  bestiary/monster book, player guide, GM guide, or other clear subtype.
• Capture edition/version info (e.g., "5e", "2nd Edition", "Pathfinder 2e", "OSR").
• If present, capture the game system or compatibility tag (e.g., "D&D 5e", "Mothership",
  "System Neutral", "Forged in the Dark", "Year Zero", "Powered by the Apocalypse").
• Populate keywords (max 10) that reflect genre/setting (fantasy, sci-fi, horror, cyberpunk,
  post-apocalyptic), tone (grimdark, heroic), format (zine, hardcover), and notable mechanics.
  Add keywords for the product type, for example core rulebook, adventure/module, supplement, 
  setting, bestiary/monster book, player guide, GM guide, or other clear subtype.
• Provide a concise description summarizing visible text (blurb, module hook, setting flavor).

Guidance:
• Prefer information printed on covers, title pages, and colophons; corroborate with OCR excerpts.
• If uncertain, return null and keep confidence near 0 rather than guessing.
"""

MAX_REQUESTS_PER_BATCH = 50_000
MAX_BATCH_BYTES = 128 * 1024 * 1024

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESOURCES_ROOT = PROJECT_ROOT / "resources"
CACHE_ROOT = PROJECT_ROOT / ".cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_PATH = RESOURCES_ROOT / "bookmeta.db"
SECRETS_PATH = PROJECT_ROOT / "secrets.json"
