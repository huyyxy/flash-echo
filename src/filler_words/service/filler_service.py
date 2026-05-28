from collections import defaultdict, deque

from filler_words.api.schemas import FillerRequest, FillerResponse
from filler_words.core.cache import LruTtlCache
from filler_words.core.labels import FillerType
from filler_words.core.normalization import normalize_query
from filler_words.inference.classifier import BaseClassifier
from filler_words.templates.registry import TemplateRegistry


class FillerService:
    def __init__(
        self,
        *,
        classifier: BaseClassifier,
        templates: TemplateRegistry,
        cache: LruTtlCache,
        recent_window_size: int = 5,
    ) -> None:
        self.classifier = classifier
        self.templates = templates
        self.cache = cache
        self.recent_window_size = recent_window_size
        self._recent_templates: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=recent_window_size))

    def decide(self, request: FillerRequest) -> FillerResponse:
        normalized_query = normalize_query(request.query)
        if len(normalized_query) <= 1:
            return self._empty_response(request)

        cache_key = f"{self.classifier.model_version}:{normalized_query}"
        result = self.cache.get(cache_key)
        if result is None:
            try:
                result = self.classifier.classify(normalized_query)
            except Exception:
                return self._empty_response(request)
            self.cache.set(cache_key, result)

        if result.filler_type is FillerType.NONE:
            return self._empty_response(request, confidence=result.confidence)

        recent_key = request.request_id or request.persona_tag
        recent_ids = set(self._recent_templates[recent_key])
        template = self.templates.choose(
            locale=request.locale,
            persona_tag=request.persona_tag,
            filler_type=result.filler_type,
            recent_template_ids=recent_ids,
        )
        if template is None:
            return self._empty_response(request, confidence=result.confidence)

        self._recent_templates[recent_key].append(template.template_id)
        return FillerResponse(
            request_id=request.request_id,
            trigger=1,
            filler_type=result.filler_type,
            filler_text=template.text,
            template_id=template.template_id,
            model_version=self.classifier.model_version,
            confidence=result.confidence,
        )

    def _empty_response(self, request: FillerRequest, confidence: float | None = None) -> FillerResponse:
        return FillerResponse(
            request_id=request.request_id,
            trigger=0,
            filler_type=FillerType.NONE,
            filler_text="",
            template_id="",
            model_version=self.classifier.model_version,
            confidence=confidence,
        )
