# Evaluation Results

Detailed evaluation results for the ORKG LLM Research Fields Classifier.

**Evaluation Date**: 2026-01-21  
**Dataset**: 15 papers from SARAG evaluation dataset  
**Data Source**: [SARAG Evaluation App](https://sarag-evaluation.netlify.app/dashboard?view=overview)

---

## Summary Metrics

| Approach | Top-1 Acc | Top-5 Acc | ORKG Score | Avg Time | Total Tokens |
|----------|-----------|-----------|------------|----------|--------------|
| **Embedding** ⭐ | 20.0% | **46.7%** | **40.0%** | **2992ms** | ~45K |
| **Single-Shot** | **20.0%** | 20.0% | 20.0% | 4980ms | ~75K |
| Top-Down | 0.0% | 20.0% | 16.0% | 5450ms | ~52K |
| Old Classifier | 0.0% | 13.3% | 9.3% | 1305ms | N/A |

### Key Findings

1. **Embedding approach achieves best overall score**: 40.0% ORKG Score with 46.7% Top-5 accuracy
2. **Embedding and Single-Shot tie for Top-1**: Both achieve 20.0% exact match rate
3. **Embedding is fastest LLM approach**: 2992ms average, best speed/accuracy balance
4. **4.3x improvement over baseline**: ORKG Score improved from 9.3% to 40.0%
5. **3.5x improvement in Top-5**: From 13.3% to 46.7%

---

## ORKG Scoring System

The ORKG accuracy score is a **position-weighted metric** that rewards correct predictions appearing higher in the ranked list:

| Position | Score | Description |
|----------|-------|-------------|
| Rank 1 | 100% | Exact match - ground truth is top prediction |
| Rank 2-3 | 80% | Close - ground truth in positions 2 or 3 |
| Rank 4-5 | 60% | Found - ground truth in positions 4 or 5 |
| Not Found | 0% | Ground truth not in top 5 predictions |

**Final Score** = Average of all paper scores

---

## Baseline Analysis (Old Classifier)

### Baseline Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **ORKG Score** | **9.3%** | Position-weighted accuracy |
| Top-1 Accuracy | 0.0% | Ground truth is #1 prediction |
| Top-5 Accuracy | 13.3% | Ground truth in top 5 |
| Not Found Rate | 86.7% | Ground truth not found |
| Avg Response Time | 1305ms | Fastest (but least accurate) |

### Position Distribution (Old Classifier)

| Position | Count | Percentage |
|----------|-------|------------|
| Rank 1 | 0 | 0.0% |
| Rank 2 | 1 | 6.7% |
| Rank 3 | 1 | 6.7% |
| Rank 4 | 0 | 0.0% |
| Rank 5 | 0 | 0.0% |
| **Not Found** | **13** | **86.7%** |

### Common Failure Patterns (Old Classifier)

| Pattern | Example | Issue |
|---------|---------|-------|
| **Method over Domain** | Soil moisture ML → "Machine Learning" instead of "Plant Cultivation" | Classifies by technique, not application |
| **Sibling Field** | CSO Classifier → "Digital Libraries" instead of "Databases" | Correct branch, wrong leaf |
| **Domain Confusion** | Breast cancer survival → "Bioinformatics" instead of "Epidemiology" | Medical domain misclassification |

---

## Improvement Over Baseline

| Metric | Baseline | Single-Shot | Embedding | Best Improvement |
|--------|----------|-------------|-----------|------------------|
| Top-1 Accuracy | 0.0% | **20.0%** | **20.0%** | **+20.0%** |
| Top-5 Accuracy | 13.3% | 20.0% | **46.7%** | **+33.4%** |
| ORKG Score | 9.3% | 20.0% | **40.0%** | **+30.7%** |
| Not Found Rate | 86.7% | ~80% | ~53% | **-33.7%** |

---

## Approach Deep Dive

### Embedding-Based Hybrid Approach ⭐ Recommended

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Abstract  │────▶│  LLM Analysis    │────▶│  Semantic   │
│             │     │  (description)   │     │  Description│
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                    │
┌─────────────┐     ┌──────────────────┐            │
│  Top-5      │◀────│  Cosine          │◀───────────┘
│  Predictions│     │  Similarity      │
└─────────────┘     └──────────────────┘
```

| Metric | Value |
|--------|-------|
| LLM Calls | 1 + embedding call |
| Prompt Size | ~3K tokens |
| Avg Response | **2992ms** |
| Top-1 Accuracy | 20.0% |
| **Top-5 Accuracy** | **46.7%** |
| **ORKG Score** | **40.0%** |

**Pros:** Fastest LLM approach, best Top-5, best ORKG Score, semantic matching  
**Cons:** Two-step process, requires embedding cache  
**Best For:** Production use, real-time classification, best overall balance

---

### Single-Shot Approach

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Abstract  │────▶│  LLM + Full      │────▶│  Top-5      │
│             │     │  Taxonomy (570)  │     │  Predictions│
└─────────────┘     └──────────────────┘     └─────────────┘
```

| Metric | Value |
|--------|-------|
| LLM Calls | 1 per paper |
| Prompt Size | ~15K tokens |
| Avg Response | 4980ms |
| **Top-1 Accuracy** | **20.0%** |
| Top-5 Accuracy | 20.0% |
| ORKG Score | 20.0% |

**Pros:** Global view, no error propagation, simple architecture  
**Cons:** Large prompts, higher cost, slower  
**Best For:** When simplicity is preferred, batch processing

---

### Top-Down Hierarchical Approach

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Abstract  │────▶│  Step 1: Select  │────▶│  Subcategory│
│             │     │  Subcategory     │     │  (e.g., CS) │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │  Step 2: Select   │
                                          │  from 3-18 fields │
                                          │  in subcategory   │
                                          └───────────────────┘
```

| Metric | Value |
|--------|-------|
| LLM Calls | 2 per paper |
| Prompt Size | ~2-3K tokens per call |
| Avg Response | 5450ms |
| Top-1 Accuracy | 0.0% |
| Top-5 Accuracy | 20.0% |
| ORKG Score | 16.0% |

**Pros:** Smaller prompts per call, structured reasoning, lower token cost  
**Cons:** Error propagation if wrong subcategory selected, slower due to 2 calls  
**Best For:** Understanding taxonomy structure, cost-sensitive applications

---

## Token Economics

| Approach | Tokens/Paper | Est. Cost (GPT-4o-mini) | Cost for 1000 Papers |
|----------|--------------|-------------------------|----------------------|
| Embedding | ~3,000 | ~$0.0005 | ~$0.50 |
| Top-Down | ~3,500 | ~$0.0006 | ~$0.60 |
| Single-Shot | ~5,000 | ~$0.0008 | ~$0.80 |
| Old Classifier | N/A | Free | Free |

*Based on GPT-4o-mini: $0.15/1M input, $0.60/1M output*

---

## Performance Comparison

### Speed vs Accuracy Trade-off

| Approach | ORKG Score | Avg Time | Score/Second |
|----------|------------|----------|--------------|
| **Embedding** | **40.0%** | **2992ms** | **13.4%/s** ⭐ |
| Single-Shot | 20.0% | 4980ms | 4.0%/s |
| Top-Down | 16.0% | 5450ms | 2.9%/s |
| Old Classifier | 9.3% | 1305ms | 7.1%/s |

### Cost vs Accuracy Trade-off

| Approach | ORKG Score | Est. Cost | Score/$ |
|----------|------------|-----------|---------|
| **Embedding** | **40.0%** | $0.0005 | **80,000** ⭐ |
| Top-Down | 16.0% | $0.0006 | 26,667 |
| Single-Shot | 20.0% | $0.0008 | 25,000 |

---

## Evaluation Charts

The visualization generates these charts in the combined report:

| Chart | Description |
|-------|-------------|
| 🎯 **Radar Chart** | Multi-dimensional comparison (accuracy, speed, cost, proximity) |
| 📊 **Rank Distribution** | Where ground truth appears in predictions per approach |
| ⏱️ **Response Time per Paper** | Performance comparison across all papers |
| ⚙️ **Timing Breakdown** | LLM processing vs overhead time |
| 💰 **Token Usage** | Input vs output tokens per approach |
| 🌳 **H-Distance Analysis** | Taxonomic proximity of predictions |

---

## Recommendations

### Best Approach by Use Case

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| **Production deployment** | **Embedding** ⭐ | Best overall (40% ORKG, fast) |
| Maximum Top-5 accuracy | **Embedding** | 46.7% in top 5 |
| Best overall score | **Embedding** | 40.0% ORKG Score |
| Fastest response | **Embedding** | 2992ms average |
| Lowest cost | **Embedding** | ~$0.0005/paper |
| Real-time classification | **Embedding** | Best speed/accuracy |
| Simple architecture | Single-Shot | Single LLM call |
| Batch processing | Single-Shot or Embedding | Both work well |

### Why Embedding Wins

1. **Best ORKG Score (40.0%)** - 2x better than Single-Shot
2. **Best Top-5 Accuracy (46.7%)** - 2.3x better than Single-Shot
3. **Fastest (2992ms)** - 40% faster than Single-Shot
4. **Lowest Cost (~$0.0005)** - 37% cheaper than Single-Shot
5. **Best Efficiency** - Highest score per second and per dollar

### Future Improvements

1. **Ensemble approach**: Combine predictions from multiple classifiers
2. **Two-stage classification**: First detect methodology vs application papers
3. **Fine-tuned prompts**: Per-domain prompt optimization
4. **Expanded dataset**: More papers for robust evaluation
5. **Confidence calibration**: Better confidence scores

---

## Running the Evaluation

```bash
# Evaluate all approaches
python evaluation/evaluation_script.py

# Evaluate specific approaches only
python evaluation/evaluation_script.py --approaches single_shot,embedding

# Limit number of papers (for testing)
python evaluation/evaluation_script.py --max-papers 5

# Use custom dataset
python evaluation/evaluation_script.py --dataset path/to/dataset.csv
```

### Generate Reports

```bash
python evaluation/visualization.py
```

---

## Dataset Format

The evaluation dataset (`evaluation/evaluation_dataset.csv`) should have:

| Column | Required | Description |
|--------|----------|-------------|
| `paper_id` | ✅ | Unique identifier (e.g., ORKG resource ID) |
| `doi` | ❌ | Paper DOI |
| `title` | ✅ | Paper title |
| `research_field_id` | ✅ | Ground truth ORKG field ID (e.g., R112118) |
| `research_field_name` | ✅ | Ground truth field name |
| `validated_url` | ❌ | URL for verification |

---

## Output Structure

```
evaluation_results/
├── single_shot/
│   └── single_shot_results_YYYYMMDD_HHMMSS.json
├── top_down/
│   └── top_down_results_YYYYMMDD_HHMMSS.json
├── embedding/
│   └── embedding_results_YYYYMMDD_HHMMSS.json
└── old_classifier/
    └── old_classifier_results_YYYYMMDD_HHMMSS.json

reports/
├── single_shot/
│   └── evaluation_report_YYYYMMDD_HHMMSS.html
├── top_down/
│   └── evaluation_report_YYYYMMDD_HHMMSS.html
├── embedding/
│   └── evaluation_report_YYYYMMDD_HHMMSS.html
└── combined/
    └── evaluation_report_YYYYMMDD_HHMMSS.html  ← Main comparison report
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Top-1 Accuracy** | Ground truth matches the #1 prediction |
| **Top-5 Accuracy** | Ground truth appears anywhere in top 5 |
| **ORKG Score** | Position-weighted accuracy (100/80/60/0) |
| **H-Distance** | Hierarchical distance in taxonomy tree (0 = exact) |
| **Response Time** | Total time for classification (ms) |
| **Token Usage** | LLM tokens consumed |

---

*Report last updated: 2026-01-21*

*See [Main README](../README.md) for project overview and setup instructions.*