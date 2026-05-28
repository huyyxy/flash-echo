import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filler_words.core.labels import FillerType


@dataclass(frozen=True)
class Template:
    template_id: str
    text: str
    enabled: bool = True
    weight: int = 1


class TemplateRegistry:
    def __init__(self, version: str, locales: dict[str, Any], default_persona: str) -> None:
        self.version = version
        self.locales = locales
        self.default_persona = default_persona

    @classmethod
    def from_file(cls, path: str | Path) -> "TemplateRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=data["version"],
            locales=data["locales"],
            default_persona=data.get("default_persona", "male_white_collar"),
        )

    def choose(
        self,
        *,
        locale: str,
        persona_tag: str,
        filler_type: FillerType,
        recent_template_ids: set[str] | None = None,
    ) -> Template | None:
        candidates = self._templates(locale, persona_tag, filler_type)
        if not candidates and persona_tag != self.default_persona:
            candidates = self._templates(locale, self.default_persona, filler_type)
        if not candidates:
            return None

        recent_template_ids = recent_template_ids or set()
        fresh = [template for template in candidates if template.template_id not in recent_template_ids]
        pool = fresh or candidates
        weights = [max(template.weight, 1) for template in pool]
        return random.choices(pool, weights=weights, k=1)[0]

    def _templates(self, locale: str, persona_tag: str, filler_type: FillerType) -> list[Template]:
        raw_templates = (
            self.locales.get(locale, {})
            .get(persona_tag, {})
            .get(filler_type.value, [])
        )
        return [
            Template(
                template_id=item["template_id"],
                text=item["text"],
                enabled=item.get("enabled", True),
                weight=item.get("weight", 1),
            )
            for item in raw_templates
            if item.get("enabled", True)
        ]
