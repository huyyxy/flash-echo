import json
import re
import sys
from pathlib import Path


BOUNDARY_RE = re.compile(r"[，。！,!]$")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/templates.zh-CN.v1.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    seen_ids: set[str] = set()

    for locale, personas in data["locales"].items():
        for persona, types in personas.items():
            for filler_type, templates in types.items():
                for item in templates:
                    template_id = item["template_id"]
                    text = item["text"]
                    if template_id in seen_ids:
                        errors.append(f"duplicate template_id: {template_id}")
                    seen_ids.add(template_id)
                    if not template_id.startswith(f"{locale}.{persona}.{filler_type.lower()}."):
                        errors.append(f"template_id does not match path: {template_id}")
                    if not BOUNDARY_RE.search(text):
                        errors.append(f"template text must end with a playable boundary: {template_id}")
                    if len(text) > 32:
                        errors.append(f"template text is too long: {template_id}")

    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"validated {len(seen_ids)} templates from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
