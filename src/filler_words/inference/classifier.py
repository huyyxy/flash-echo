from abc import ABC, abstractmethod

from filler_words.core.cache import ClassificationResult
from filler_words.core.labels import FillerType


class BaseClassifier(ABC):
    model_version: str

    @abstractmethod
    def classify(self, query: str) -> ClassificationResult:
        raise NotImplementedError


class HeuristicClassifier(BaseClassifier):
    """Conservative bootstrap classifier used until the ONNX package is available."""

    model_version = "heuristic-bootstrap-v0"

    _none_queries = ("你好", "您好", "再见", "拜拜", "谢谢", "好的", "嗯", "一加一等于几")
    _empathy_terms = ("焦虑", "难过", "压力", "失眠", "崩溃", "害怕", "担心", "烦")
    _retrieval_terms = ("查", "找", "搜索", "会议", "上次", "之前", "总结", "资料")
    _clarify_terms = ("这个", "那个", "怎么弄", "怎么办", "这件事")
    _ack_terms = ("帮我", "给我", "请你", "麻烦", "写一封", "规划", "生成", "整理")
    _frame_terms = ("怎么看", "如何评价", "影响", "方案", "规划一下", "设计")
    _thinking_terms = ("为什么", "怎么理解", "原因", "区别", "比较", "计算")

    def classify(self, query: str) -> ClassificationResult:
        if len(query) <= 2 or query in self._none_queries:
            return ClassificationResult(FillerType.NONE, 0.9)
        if self._contains(query, self._empathy_terms):
            return ClassificationResult(FillerType.EMPATHY, 0.72)
        if self._contains(query, self._clarify_terms) and len(query) <= 10:
            return ClassificationResult(FillerType.CLARIFY_LEADIN, 0.68)
        if self._contains(query, self._retrieval_terms):
            return ClassificationResult(FillerType.RETRIEVAL, 0.7)
        if self._contains(query, self._ack_terms):
            return ClassificationResult(FillerType.ACKNOWLEDGE, 0.69)
        if self._contains(query, self._frame_terms):
            return ClassificationResult(FillerType.FRAME, 0.68)
        if self._contains(query, self._thinking_terms) or len(query) >= 18:
            return ClassificationResult(FillerType.THINKING, 0.64)
        return ClassificationResult(FillerType.NONE, 0.75)

    @staticmethod
    def _contains(query: str, terms: tuple[str, ...]) -> bool:
        return any(term in query for term in terms)
