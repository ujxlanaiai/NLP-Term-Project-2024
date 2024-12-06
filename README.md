# NLP-Term-Project-2024

This project leverages Upstage's LLM to improve question-answering performance through simple prompting and Retrieval-Augmented Generation (RAG) techniques, without the need for fine-tuning.

---

## Table of Contents

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)

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

---

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ujxlanaiai/NLP-Term-Project-2024.git
   cd NLP-Term-Project-2024

2. Install required dependencies
   ```bash
   pip install -r requirements.txt

---

## Usage

#### 1. Create Document Embeddings for `ewha.pdf`
Follow these steps to preprocess and embed the documents:

1. Open and run the notebook: `docs/preprocess_txt_.ipynb`.  
   - **Description**: Prepares the text data for embedding.

2. Open and run the notebook: `docs/embed_json.ipynb`.  
   - **Description**: Generates document embeddings in JSON format.

3. Verify that the JSON files are correctly generated in the `docs/` directory:  
   - Ensure the following files exist:
     - `embedded_doc0.json`
     - `embedded_doc1.json`
     - `embedded_doc2.json`
     - `embedded_title.json`

---

#### 2. Run the Main Notebook
To evaluate the results and calculate the final accuracy:

1. Open the notebook: `main.ipynb` in your preferred Jupyter environment.  
2. Execute all cells to run the final pipeline and display the output accuracy.

