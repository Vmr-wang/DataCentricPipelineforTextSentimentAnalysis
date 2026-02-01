# A Pipeline for Text Sentiment Analysis

This work is our course project for Data Preparation(5284DAPR6Y) in University of Amsterdam.

We build a flexible machine learning pipeline for sentiment analysis on **any** review dataset in JSON or JSONL format. No hardcoded dataset assumptions - you specify the columns!

And the code in **MakeDirty** is to turn clean datasets into "dirty" datasets(with errors).

You can use requirments.txt file to build the environment.

## 🌟 Key Features

- **Works with ANY review dataset** - Just specify your column names
- **No hardcoded assumptions** - Flexible column mapping
- **Smart data cleaning** - Optional and configurable
- **Command-line interface** - Easy to use
- **Production-ready** - Comprehensive error handling

## 📦 What Makes This Generic?

Unlike traditional pipelines that only work with specific datasets (Goodreads, Amazon, etc.), this pipeline:

✅ Accepts **any JSON or JSONL file**
✅ Lets **you specify** which columns to use
✅ Doesn't assume specific column names
✅ Handles various data formats automatically

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python main.py \
    --data-path your_reviews.json \
    --text-column review_text \
    --label-column rating
```

That's it! The pipeline will:
1. Load your data
2. Clean it (if enabled)
3. Train Word2Vec on the text
4. Train XGBoost classifier
5. Evaluate and visualize results

## 📋 Required Arguments

Only **3 arguments** are required:

| Argument | Description | Example |
|----------|-------------|---------|
| `--data-path` | Path to your data file | `reviews.json` |
| `--text-column` | Column with review text | `review_text` |
| `--label-column` | Column with labels/ratings | `rating` |

Everything else has sensible defaults!

## 📊 Supported Data Formats

### JSON Format

```json
[
  {
    "review_text": "This product is great!",
    "rating": 5,
    "helpful_votes": 10
  },
  {
    "review_text": "Not bad, could be better.",
    "rating": 3,
    "helpful_votes": 2
  }
]
```

### JSONL Format

```jsonl
{"text": "Amazing!", "score": 5, "verified": true}
{"text": "Terrible product", "score": 1, "verified": false}
```

### Nested JSON

```json
{
  "data": [
    {"content": "Great!", "sentiment": "positive"},
    {"content": "Bad!", "sentiment": "negative"}
  ]
}
```

The pipeline automatically handles all these formats!

## 💡 Usage Examples

### Example 1: Goodreads Reviews

```bash
python main.py \
    --data-path goodreads_reviews.json \
    --text-column review_text \
    --label-column rating \
    --cleaning all
```

### Example 2: Amazon Reviews (with additional features)

```bash
python main.py \
    --data-path amazon_reviews.jsonl \
    --text-column text \
    --label-column rating \
    --additional-features title
```

**Note**: Use `--additional-features` followed by **space-separated** column names for multiple features!

### Example 3: Yelp Reviews

```bash
python main.py \
    --data-path yelp_reviews.json \
    --text-column text \
    --label-column stars \
    --cleaning all
```

### Example 4:

```bash
python main.py \
    --data-path my_custom_reviews.jsonl \
    --text-column review_content \
    --label-column sentiment_score \
    --vector-size 200 \
    --n-estimators 300
```

## 🛠️ All Command Line Options

### Required

```bash
--data-path PATH              # Path to your JSON/JSONL file
--text-column NAME            # Column name for review text
--label-column NAME           # Column name for labels/ratings
```

### Optional Data Configuration

```bash
--output-path PATH                    # Where to save results (default: ./outputs/results.png)
--additional-features F1 F2 F3 ...    # Extra numeric columns (space-separated)
```

### Data Cleaning

```bash
--cleaning OPTION [OPTION ...]
```

Options:
- `all` - Enable all cleaning (default)
- `none` - Disable all cleaning
- `missing_values` - Remove rows with missing data
- `duplicates` - Remove duplicate rows
- `text_garbling` - Fix text issues (spaces, special chars)
- `type_errors` - Fix type inconsistencies
- `remove_empty_text` - Remove empty text rows

### Model Parameters

**Data Splitting:**
```bash
--test-size 0.2              # Proportion for test set (default: 0.2)
--random-state 42            # Random seed (default: 42)
```

**Word2Vec:**
```bash
--vector-size 300            # Vector dimension (default: 300)
--window 5                   # Context window (default: 5)
--min-count 2                # Min word frequency (default: 2)
```

**XGBoost:**
```bash
--n-estimators 200           # Number of trees (default: 200)
--max-depth 6                # Max tree depth (default: 6)
--learning-rate 0.1          # Learning rate (default: 0.1)
```

**Cross-Validation:**
```bash
--n-folds 5                  # Number of CV folds (default: 5)
```

**Output:**
```bash
--quiet                      # Suppress detailed output
```


### Visualization

The output PNG contains:
1. **Confusion Matrix** - See where the model makes mistakes
2. **Class Distribution** - Understand label balance
3. **CV Scores** - Assess model stability


## 📄 License

MIT License - Free to use, modify, and distribute

---

**Remember**: This pipeline is **dataset-agnostic**. As long as you have review text and labels in JSON/JSONL format, it will work!
