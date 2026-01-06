# Kharagpur Data Science Hackathon - Track A
## Novel-Backstory Consistency Classification System

A research-grade Python pipeline that determines whether a hypothetical character backstory is logically and causally consistent with a full-length novel.

### Task Description

This is a **binary classification task**:
- `1` → backstory is consistent with the novel
- `0` → backstory contradicts the novel

The system focuses on **global consistency, constraint tracking, and evidence-based reasoning** using semantic retrieval and deterministic aggregation rules.

### Technical Architecture

The pipeline consists of 5 main components:

1. **Ingestion & Chunking**: Loads novels and chunks them into 800-1200 token segments
2. **Vector Store**: Uses Pathway for semantic indexing with sentence transformers
3. **Retrieval**: Retrieves top-K relevant chunks using backstory as semantic query
4. **Reasoning**: Evaluates chunks as SUPPORT/CONTRADICT/NEUTRAL using linguistic signals
5. **Aggregation**: Applies deterministic rules for binary classification

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Basic Usage
```bash
python main.py --input data/test.csv --output results.csv
```

#### Advanced Usage
```bash
python main.py \
    --input data/test.csv \
    --output results.csv \
    --novels-dir data/novel \
    --model all-MiniLM-L6-v2
```

#### Parameters
- `--input`: Input CSV file with backstories (default: data/test.csv)
- `--output`: Output CSV file with predictions (default: results.csv)
- `--novels-dir`: Directory containing novel .txt files (default: data/novel)
- `--model`: Sentence transformer model name (default: all-MiniLM-L6-v2)

### Dataset Structure

```
data/
├── novel/
│   ├── In search of the castaways.txt
│   └── The Count of Monte Cristo.txt
├── train.csv
└── test.csv
```

#### CSV Format
- `id`: Unique identifier
- `book_name`: Name of the novel
- `char`: Character name
- `caption`: Optional caption
- `content`: Backstory text
- `label`: Ground truth label (train.csv only)

### Pipeline Details

#### 1. Text Chunking
- Novels are chunked into 800-1200 token segments
- Overlapping chunks with 200 token overlap
- Maintains paragraph coherence

#### 2. Semantic Retrieval
- Uses sentence transformers for embeddings
- Cosine similarity for relevance scoring
- Top-K retrieval with similarity thresholding

#### 3. Consistency Reasoning
- **SUPPORT**: Entity overlap, supporting keywords, semantic similarity
- **CONTRADICT**: Negation patterns, contradiction keywords, entity conflicts
- **NEUTRAL**: Neither strong support nor contradiction

#### 4. Deterministic Aggregation
- **Primary Rule**: Any strong contradiction → `0` (inconsistent)
- **Default Rule**: No strong contradictions → `1` (consistent)
- Contradiction threshold: 0.4 confidence
- Support threshold: 0.3 confidence

### Output Files

#### results.csv
```csv
id,prediction
95,1
136,0
59,1
...
```

#### results_detailed.csv
Contains full reasoning, evaluations, and retrieved chunks for analysis.

### Performance Considerations

- **Memory**: Vector embeddings stored in memory
- **Speed**: Batch processing for multiple backstories
- **Scalability**: Pathway handles large text collections efficiently

### Dependencies

- `pathway>=0.12.0`: Vector indexing and processing
- `sentence-transformers>=2.2.2`: Semantic embeddings
- `numpy>=1.21.0`: Numerical operations
- `pandas>=1.3.0`: Data manipulation
- `tiktoken>=0.5.0`: Token counting
- `scikit-learn>=1.0.0`: Machine learning utilities

### Code Structure

```
src/
├── __init__.py
├── ingestion.py      # Novel loading and chunking
├── vector_store.py   # Semantic indexing with Pathway
├── retrieval.py      # Semantic search and retrieval
├── reasoning.py      # SUPPORT/CONTRADICT/NEUTRAL evaluation
└── aggregation.py    # Binary classification rules
```

### Example Output

```
Starting consistency classification pipeline...
Setting up vector store...
Processing novel: In search of the castaways
Generated 234 chunks from In search of the castaways
Processing novel: The Count of Monte Cristo
Generated 567 chunks from The Count of Monte Cristo
Generated 801 total chunks
Creating embeddings...
Vector store setup complete!

Processing 60 backstories...
Retrieving relevant chunks...
Evaluating consistency...
Applying aggregation rules...
Saving results to results.csv

==================================================
PIPELINE STATISTICS
==================================================
Total predictions: 60
Consistent (1): 42
Inconsistent (0): 18
Consistent rate: 0.700

Retrieval Statistics:
Average chunks per query: 7.2
Max chunks: 10
Min chunks: 3
Average similarity: 0.456

Classification Statistics:
Average support count: 2.1
Average contradiction count: 0.8
Average neutral count: 4.3
```

### License

This project is developed for the Kharagpur Data Science Hackathon 2026.
