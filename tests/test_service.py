from filler_words.api.schemas import FillerRequest
from filler_words.app import build_service
from filler_words.core.labels import FillerType


def test_simple_query_does_not_trigger() -> None:
    service = build_service()

    response = service.decide(FillerRequest(query="你好", persona_tag="male_white_collar"))

    assert response.trigger == 0
    assert response.filler_type is FillerType.NONE
    assert response.filler_text == ""
    assert response.template_id == ""


def test_open_question_returns_persona_template() -> None:
    service = build_service()

    response = service.decide(FillerRequest(query="你怎么看 AI 对教育行业的影响？", persona_tag="male_white_collar"))

    assert response.trigger == 1
    assert response.filler_type is FillerType.FRAME
    assert response.filler_text
    assert response.template_id.startswith("zh-CN.male_white_collar.frame.")


def test_unknown_persona_falls_back_to_default_persona() -> None:
    service = build_service()

    response = service.decide(FillerRequest(query="帮我写一封请假邮件", persona_tag="unknown"))

    assert response.trigger == 1
    assert response.filler_type is FillerType.ACKNOWLEDGE
    assert response.template_id.startswith("zh-CN.male_white_collar.acknowledge.")
