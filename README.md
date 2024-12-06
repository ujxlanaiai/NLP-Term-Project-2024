# NLP-Term-Project-2024

This project leverages Upstage's LLM to improve question-answering performance through simple prompting and Retrieval-Augmented Generation (RAG) techniques, without the need for fine-tuning.

---

## Table of Contents

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

---

## Directory Structure

```
.
├── mmlu.ipynb                 # mmlu module file
├── testset.csv                # Dataset file
├── ewha/                      # ewha.pdf module directory
│   ├── __init__.py          
│   ├── ewha_run.py            # Main execution script for ewha module
│   ├── tests/                 # Directory containing various test notebooks
│   ├── llms/                  # Directory for LLM-related modules
│   │   ├── block_A.py
│   │   ├── block_B.py
│   │   ├── block_C.py
│   │   ├── block_D.py
│   │   ├── block_E.py
│   │   ├── block_F.py
│   │   ├── __init__.py
│   │   ├── utils.py           # Utility functions
│   │   ├── experiment/        # Directory for experimental notebooks
├── docs/                      # Directory for documentation and preprocessing
│   ├── doc_0.json
│   ├── doc_1.json
│   ├── doc_2.json
│   ├── embedded_doc0.json
│   ├── embedded_doc1.json
│   ├── embedded_doc2.json
│   ├── embedded_title.json
│   ├── preprocess_txt_.ipynb
│   └── embed_json.ipynb
└── main.ipynb                 # Main notebook for the project
```

