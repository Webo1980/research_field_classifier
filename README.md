# ORKG LLM Research Field Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An LLM-powered multi-approach research paper classification system for the [Open Research Knowledge Graph (ORKG)](https://orkg.org). This classifier assigns research papers to appropriate research fields from the ORKG taxonomy using three distinct approaches.

## 🎯 Overview

This project provides an improved research field classification system that leverages Large Language Models (LLMs) to classify scientific papers into ORKG research fields. It offers three classification approaches, each with different trade-offs between accuracy, cost, and speed.

### Comparison with Previous Classifier

| Feature | Old Classifier | This Classifier |
|---------|---------------|-----------------|
| Approach | Single flat ML model | Multiple LLM-based approaches |
| Taxonomy awareness | Limited | Full hierarchical understanding |
| ORKG Score | 9.3% | **40.0%** (4.3x improvement) |
| Top-1 Accuracy | 0% | **20%** |
| Top-5 Accuracy | 13.3% | **46.7%** (3.5x improvement) |
| Explainability | Score only | Reasoning + taxonomy paths |
| Adaptability | Requires retraining | Prompt-based, easily updated |

## 📊 Classification Approaches

| Approach | Description | Top-1 Acc | Top-5 Acc | ORKG Score | Avg Time | Best For |
|----------|-------------|-----------|-----------|------------|----------|----------|
| **Embedding** | LLM analysis + semantic matching | 20.0% | **46.7%** | **40.0%** | ~3.0s | ⭐ Best overall |
| **Single-Shot** | Full taxonomy to LLM in one call | **20.0%** | 20.0% | 20.0% | ~5.0s | Maximum Top-1 |
| **Top-Down** | Hierarchical navigation | 0.0% | 20.0% | 16.0% | ~5.5s | Cost-sensitive |
| Old Classifier | Flat ML baseline | 0.0% | 13.3% | 9.3% | ~1.3s | Legacy comparison |

*Results based on evaluation with 15 papers from SARAG dataset (https://sarag-evaluation.netlify.app/dashboard?view=overview) (January 2026)*

### ORKG Scoring System

The ORKG Score is a position-weighted accuracy metric:

| Position | Score | Description |
|----------|-------|-------------|
| Rank 1 | 100% | Exact match - ground truth is top prediction |
| Rank 2-3 | 80% | Close - ground truth in positions 2 or 3 |
| Rank 4-5 | 60% | Found - ground truth in positions 4 or 5 |
| Not Found | 0% | Ground truth not in top 5 predictions |

### Approach Details

#### 1. Embedding-Based Hybrid Approach ⭐ Recommended
Uses LLM for semantic understanding combined with pre-computed taxonomy embeddings.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Paper Abstract │────▶│  LLM Analysis   │────▶│  Field          │
│                 │     │  (2-3 sentences)│     │  Description    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│  Top-N Similar  │◀────│  Embedding      │◀─────────────┘
│  Fields         │     │  Matching       │
└─────────────────┘     └─────────────────┘
```

**Performance:** 20% Top-1 | **46.7% Top-5** | **40% ORKG Score** | ~3s  
**Pros:** Fast, best Top-5 accuracy, semantic similarity matching  
**Cons:** Two-step process, requires embedding cache  
**Best For:** Production use, best speed/accuracy balance

#### 2. Single-Shot Approach
Provides the complete ORKG taxonomy (~570 fields) to the LLM in a single prompt.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Paper Abstract │────▶│  LLM + Full     │────▶│  Top-N Fields   │
│                 │     │  Taxonomy       │     │  with Paths     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Performance:** **20% Top-1** | 20% Top-5 | 20% ORKG Score | ~5s  
**Pros:** Global view of all fields, no error propagation, simple architecture  
**Cons:** Large prompts (~15K tokens), higher cost  
**Best For:** When Top-1 accuracy is critical

#### 3. Top-Down Hierarchical Approach
Navigates the taxonomy tree by first selecting a subcategory, then selecting fields within it.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Paper Abstract │────▶│  Step 1: Select │────▶│  Subcategory    │
│                 │     │  Subcategory    │     │  (e.g., CS)     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                               ┌─────────▼─────────┐
                                               │  Step 2: Select   │
                                               │  from 3-18 fields │
                                               │  in subcategory   │
                                               └───────────────────┘
```

**Performance:** 0% Top-1 | 20% Top-5 | 16% ORKG Score | ~5.5s  
**Pros:** Smaller prompts per call, lower token cost, structured reasoning  
**Cons:** Potential error propagation if wrong subcategory selected  
**Best For:** Understanding taxonomy navigation, cost-sensitive applications

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (or Mistral/Anthropic)

### Installation

```bash
# Clone the repository
git clone https://github.com/Webo1980/orkg-llm-research-fields-classifier.git
cd orkg-llm-research-field-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
# LLM Provider (openai, mistral, anthropic)
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini

# Embedding Settings (for embedding approach)
EMBEDDING_MODEL=text-embedding-3-small

# Cache Settings
TAXONOMY_CACHE_TTL_HOURS=168  # 1 week
EMBEDDINGS_CACHE_TTL_HOURS=168

# Default approach for /annotator endpoints
DEFAULT_APPROACH=embedding
```

### Run the API

```bash
python main.py
```

The API will be available at `http://localhost:8080`.

## 📖 API Usage

### Classify with Specific Approach

```bash
# Embedding approach (recommended)
curl -X POST "http://localhost:8080/classify/embedding" \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "Deep learning for COVID-19 detection from CT images", "top_n": 5}'

# Single-shot approach
curl -X POST "http://localhost:8080/classify/single_shot" \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "Your paper abstract here", "top_n": 5}'

# Top-down approach
curl -X POST "http://localhost:8080/classify/top_down" \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "Your paper abstract here", "top_n": 5}'
```

### Classify with All Approaches (Parallel)

```bash
curl -X POST "http://localhost:8080/classify/all" \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "Your paper abstract here", "top_n": 5}'
```

### Response Format

```json
{
  "timestamp": "2026-01-21T22:38:57.229613",
  "uuid": "d7d2e3c7-9598-4aa1-9d05-510383d2eba2",
  "approach": "embedding",
  "payload": {
    "annotations": [
      {
        "research_field": "Computer Vision and Pattern Recognition",
        "research_field_id": "R112118",
        "score": 0.95,
        "reasoning": "The paper focuses on image analysis using deep learning for medical CT images.",
        "path": ["Research Field", "Physical Sciences & Mathematics", "Computer Sciences", "Computer Vision and Pattern Recognition"],
        "path_ids": ["R112118"]
      }
    ]
  },
  "timing": {
    "total_time_ms": 2992.82,
    "llm_time_ms": 2500.53,
    "taxonomy_lookup_ms": 0.29
  },
  "token_usage": {
    "input_tokens": 3000,
    "output_tokens": 168,
    "total_tokens": 3168
  }
}
```

### Compatible with Old API

```bash
# Simple endpoint (uses default approach - embedding)
curl -X POST "http://localhost:8080/annotator/research-fields" \
  -H "Content-Type: application/json" \
  -d '{"raw_input": "Your paper abstract here"}'
```

## 📁 Project Structure

```
orkg-llm-research-field-classifier/
├── config.py                    # Configuration settings
├── main.py                      # FastAPI application
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── README.md                    # This file
│
├── services/
│   ├── __init__.py              # Service factory
│   ├── base/                    # Shared base services
│   │   ├── taxonomy_service.py  # Dynamic taxonomy management
│   │   ├── llm_service.py       # LLM API abstraction
│   │   ├── embedding_service.py # Embedding generation & caching
│   │   ├── orkg_api_service.py  # ORKG API client
│   │   └── base_classifier.py   # Classifier interface
│   └── approaches/              # Classification approaches
│       ├── single_shot/         # Single-shot classifier
│       ├── top_down/            # Top-down hierarchical classifier
│       └── embedding/           # Embedding-based classifier
│
├── evaluation/
│   ├── README.md                # Detailed evaluation documentation
│   ├── evaluation_script.py     # Evaluate all approaches
│   ├── visualization.py         # Generate HTML reports
│   └── evaluation_dataset.csv   # Ground truth dataset
│
├── evaluation_results/          # Evaluation output (per approach)
├── reports/                     # HTML reports with charts
│   ├── single_shot/
│   ├── top_down/
│   ├── embedding/
│   └── combined/                # Multi-approach comparison
│
├── cache/                       # Cached data
│   ├── taxonomy_hierarchy.json  # Full ORKG taxonomy
│   ├── taxonomy_tree.txt        # Compact taxonomy format
│   ├── taxonomy_embeddings.npz  # Pre-computed embeddings
│   └── taxonomy_embeddings_meta.json  # Cache metadata
│
└── tests/                       # Unit tests
```

## 🔧 Dynamic Taxonomy & Embedding Management

The system automatically manages taxonomy and embeddings:

### Taxonomy Fetching
- **On first run:** Fetches complete taxonomy from ORKG API (~570 research fields)
- **Cache TTL:** Refreshes when cache expires (default: 1 week)
- **Fallback:** Uses existing cache if API is unavailable

### Embedding Generation
- **On first run:** Generates embeddings for all research fields
- **Cache validation:** Checks taxonomy hash, TTL, and embedding model
- **Auto-regeneration:** When taxonomy changes or cache expires

```
First Run:
1. Fetch taxonomy from ORKG API → cache/taxonomy_hierarchy.json
2. Generate compact tree → cache/taxonomy_tree.txt
3. Compute embeddings → cache/taxonomy_embeddings.npz
4. Save metadata → cache/taxonomy_embeddings_meta.json

Subsequent Runs:
1. Check cache TTL and taxonomy hash
2. If valid → Load from cache (fast)
3. If invalid → Regenerate
```

## 📊 Evaluation

> **📋 For detailed evaluation results, methodology, and analysis, see [Evaluation Documentation](evaluation/evaluation_README.md)**

### Run Evaluation

```bash
# Evaluate all approaches
python evaluation/evaluation_script.py

# Evaluate specific approaches
python evaluation/evaluation_script.py --approaches single_shot,embedding

# Limit number of papers
python evaluation/evaluation_script.py --max-papers 15
```

### Generate Reports

```bash
python evaluation/visualization.py
```

This generates comprehensive HTML reports in `reports/` with:

- **Summary Comparison Table** - All approaches side-by-side
- **Radar Chart** - Multi-dimensional comparison (accuracy, speed, cost, proximity)
- **Rank Distribution** - Where ground truth appears in predictions
- **Response Time per Paper** - Performance visualization
- **Timing Breakdown** - LLM time vs overhead
- **Token Usage Analysis** - Cost estimation
- **H-Distance Analysis** - Taxonomic proximity
- **Per-Paper Details** - Full predictions for each paper

Generated reports are available at:
- `reports/combined/evaluation_report_*.html` - **Main comparison report**
- `reports/single_shot/evaluation_report_*.html` - Single-shot detailed report
- `reports/embedding/evaluation_report_*.html` - Embedding detailed report
- `reports/top_down/evaluation_report_*.html` - Top-down detailed report

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | Ground truth is the #1 prediction |
| **Top-5 Accuracy** | Ground truth in top 5 predictions |
| **ORKG Score** | Position-weighted: 100% (R1), 80% (R2-3), 60% (R4-5), 0% (not found) |
| **H-Distance** | Hierarchical distance in taxonomy (0 = exact match) |
| **Response Time** | Total classification time |
| **Token Usage** | LLM tokens consumed (for cost estimation) |

### Latest Evaluation Results (January 2026)

| Approach | Top-1 | Top-5 | ORKG Score | Avg Time | Tokens/Paper |
|----------|-------|-------|------------|----------|--------------|
| **Embedding** | 20.0% | **46.7%** | **40.0%** | **2992ms** | ~3,000 |
| Single-Shot | **20.0%** | 20.0% | 20.0% | 4980ms | ~11,000 |
| Top-Down | 0.0% | 20.0% | 16.0% | 5450ms | ~3,500 |
| Old Baseline | 0.0% | 13.3% | 9.3% | 1305ms | N/A |

## 💰 Cost Estimation

| Approach | Tokens/Paper | Est. Cost (GPT-4o-mini) | Cost for 1000 Papers |
|----------|--------------|-------------------------|----------------------|
| Embedding | ~3,000 | ~$0.0005 | ~$0.50 |
| Top-Down | ~3,500 | ~$0.0006 | ~$0.60 |
| Single-Shot | ~11,000 | ~$0.0017 | ~$1.70 |

*Based on GPT-4o-mini pricing ($0.15/1M input, $0.60/1M output)*

## 🔌 Extending the System

### Add a New Approach

1. Create a new directory in `services/approaches/`
2. Implement the `BaseClassifier` interface:

```python
from services.base.base_classifier import BaseClassifier

class MyClassifier(BaseClassifier):
    @property
    def approach_name(self) -> str:
        return "my_approach"
    
    async def initialize(self):
        # Setup code
        pass
    
    async def classify(self, abstract: str, top_n: int = 5) -> Dict[str, Any]:
        # Classification logic
        return {
            "annotations": [...],
            "timing": {...},
            "token_usage": {...},
            "metadata": {...}
        }
```

3. Register in `services/__init__.py`

### Use Different LLM Providers

Configure in `.env`:

```env
# OpenAI (default)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Mistral
LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-large-latest

# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

## 📚 Related Resources

- **ORKG Platform:** https://orkg.org
- **ORKG API Documentation:** https://orkg.org/api-docs
- **Research Fields Taxonomy:** https://orkg.org/research-fields
- **Original Classifier:** https://gitlab.com/TIBHannover/orkg/nlp/experiments/orkg-research-fields-classifier

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Merge Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this classifier in your research, please cite:

```bibtex
@software{orkg_llm_rf_classifier,
  title = {ORKG LLM Research Field Classifier},
  author = {TIB Hannover},
  year = {2026},
  url = {https://github.com/Webo1980/research_field_classifier}
}
```

## 📧 Contact

- **TIB Hannover - ORKG Team**
- **Issues:** Please use GitHub Issues for bug reports and feature requests

---

*Built with ❤️ for the Open Research Knowledge Graph*