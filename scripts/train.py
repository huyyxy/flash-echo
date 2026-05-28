"""Training entrypoint placeholder.

Expected inputs:
- JSONL train/validation/test splits under data/processed/
- Label mapping matching filler_words.core.labels.FillerType

The v1 implementation should fine-tune hfl/rbt3 as a seven-class classifier:
NONE + ACKNOWLEDGE + THINKING + FRAME + EMPATHY + RETRIEVAL + CLARIFY_LEADIN.
"""


def main() -> int:
    raise SystemExit("training pipeline is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
