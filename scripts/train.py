"""训练入口占位脚本。

期望输入：
- `data/processed/` 下的 JSONL 训练/验证/测试切分数据
- 与 `filler_words.core.labels.FillerType` 对齐的标签映射

v1 实现应基于 `hfl/rbt3` 微调一个七分类模型：
NONE + ACKNOWLEDGE + THINKING + FRAME + EMPATHY + RETRIEVAL + CLARIFY_LEADIN。
"""


def main() -> int:
    raise SystemExit("training pipeline is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
