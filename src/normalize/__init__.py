import re


CORP_SUFFIX_PATTERN = re.compile(
    r"\b(?:incorporated|inc|llc|ltd|limited|ltda|ab|srl|sas|company|co|gmbh|sa|bv|oy|kg|kk|publishing|press|"
    r"productions|production|entertainment)\.?\b"
)
LICENSE_SPLIT_PATTERN = re.compile(
    r"(under license|produced under license|published under|distributed by|in association with).*$"
)
LEADING_ARTICLE_PATTERN = re.compile(r"^a\s+(.+)")
