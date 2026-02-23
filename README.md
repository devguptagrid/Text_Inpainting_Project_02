## Dataset

We use WikiText-2 (raw version) from HuggingFace.

Original dataset sizes:
- Train: 36,718
- Validation: 3,760
- Test: 4,358

After cleaning (removing empty lines and very short sequences):
- Train: 23,547
- Validation: 2,454
- Test: 2,850

Cleaning removes empty lines, section headers, and very short fragments to improve training stability and quality.