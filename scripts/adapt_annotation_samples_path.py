from pathlib import Path
import re
import logging

# E.g.:
# /var/work/data/wiris-math-online-incomplete/ -> ./


def adapt_annotation_samples_path(annotation_file_path: Path) -> None:
    regex = r"\/var\/work\/data\/[^\/]+"
    subst = "."

    try:
        with annotation_file_path.open(mode="r+", encoding="utf-8") as f:
            content = f.read()
            f.seek(0)
            f.truncate(0)
            f.write(re.sub(regex, subst, content, 0, re.MULTILINE))

    except Exception as e:
        logging.error(e)


# print(Path().cwd())
adapt_annotation_samples_path(
    Path("../data/strokes/wiris-math-online-incomplete/train1000.txt")
)
