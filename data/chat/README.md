# Chat Smalltalk Dataset

ShareGPT-style direct-answer smalltalk dataset for routing SFT negative samples.

- `train.jsonl`: 119,763 direct-answer smalltalk training samples
- `valid.jsonl`: 16,374 validation samples
- `test.jsonl`: 16,358 held-out test samples

Each JSONL record has exactly two turns in `conversations`: a spoken-style user query and a complete assistant reply. Assistant replies intentionally do not include `<CONTINUE_MAIN>`.

The dataset is intended to cover greetings, thanks, closings, short acknowledgements, light emotional support, simple weather/food chat, identity questions, and very small common-sense or arithmetic questions. User turns keep ASR-style oral markers such as `嗯`, `呃`, `那个`, `啊`, `呀`, and `呢`, while avoiding task requests that should route to the filler-prefix path.

The split sizes are matched to `data/filler_prefix/small_male/` at roughly a 4:6 chat-to-filler ratio. In other words, each chat split is about two thirds of the corresponding `small_male` split, so the merged routing dataset keeps direct-answer samples slightly below filler-route samples.
