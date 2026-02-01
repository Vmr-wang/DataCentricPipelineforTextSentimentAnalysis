# Generic Review Analysis Pipeline

A flexible machine learning pipeline for sentiment analysis on **any** review dataset in JSON or JSONL format. No hardcoded dataset assumptions - you specify the columns!

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
    --additional-features helpful_vote verified_purchase
```

**Note**: Use `--additional-features` followed by **space-separated** column names for multiple numeric features!

### Example 3: Yelp Reviews

```bash
python main.py \
    --data-path yelp_reviews.json \
    --text-column text \
    --label-column stars \
    --cleaning all
```

### Example 4: Custom Dataset

```bash
python main.py \
    --data-path my_custom_reviews.jsonl \
    --text-column review_content \
    --label-column sentiment_score \
    --vector-size 200 \
    --n-estimators 300
```

### Example 5: IMDB Reviews

```bash
python main.py \
    --data-path imdb_reviews.json \
    --text-column review \
    --label-column sentiment \
    --cleaning all
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

**Multiple features example**:
```bash
python main.py \
    --data-path reviews.json \
    --text-column text \
    --label-column rating \
    --additional-features helpful_votes verified_purchase quality_rating
```

💡 **See `MULTIPLE_FEATURES.md` for a detailed guide on using multiple features!**

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

## 🎯 Real-World Examples

### Movie Reviews (IMDB Style)

```bash
python main.py \
    --data-path movie_reviews.json \
    --text-column review \
    --label-column sentiment \
    --cleaning all
```

Expected data format:
```json
[
  {"review": "Amazing movie!", "sentiment": "positive"},
  {"review": "Waste of time", "sentiment": "negative"}
]
```

### Product Reviews (Amazon Style)

```bash
python main.py \
    --data-path products.jsonl \
    --text-column reviewText \
    --label-column overall \
    --additional-features helpful unixReviewTime
```

Expected data format:
```jsonl
{"reviewText": "Love it!", "overall": 5.0, "helpful": [2, 3]}
{"reviewText": "Broke after 1 day", "overall": 1.0, "helpful": [5, 5]}
```

### App Reviews

```bash
python main.py \
    --data-path app_reviews.json \
    --text-column content \
    --label-column score \
    --cleaning all
```

Expected data format:
```json
[
  {"content": "Best app ever", "score": 5},
  {"content": "Crashes constantly", "score": 1}
]
```

### Restaurant Reviews (Yelp Style)

```bash
python main.py \
    --data-path restaurants.json \
    --text-column text \
    --label-column stars \
    --additional-features useful funny cool
```

## 📈 Understanding Output

### Console Output

```
============================================================
CONFIGURATION PARAMETERS
============================================================
Data Path: your_reviews.json
Text Column: review_text
Label Column: rating
...

============================================================
DATASET STATISTICS
============================================================
Total samples: 10000
Label distribution:
  1:  1000 (10.00%)
  2:  1500 (15.00%)
  3:  2000 (20.00%)
  4:  2500 (25.00%)
  5:  3000 (30.00%)

============================================================
FINAL EXPERIMENT SUMMARY
============================================================
CV Mean Accuracy: 0.8524 (85.24%)
Test Accuracy: 0.8498 (84.98%)
Test F1-Score: 0.8456
```

### Visualization

The output PNG contains:
1. **Confusion Matrix** - See where the model makes mistakes
2. **Class Distribution** - Understand label balance
3. **CV Scores** - Assess model stability

## 🔧 Troubleshooting

### Error: "Column 'X' not found"

**Cause**: Column name doesn't match your data

**Solution**: Check your column names
```bash
# First, check what columns your data has
import pandas as pd
df = pd.read_json('your_data.json')
print(df.columns)

# Then use the correct names
python main.py --data-path your_data.json \
    --text-column actual_text_column_name \
    --label-column actual_label_column_name
```

### Error: "Data file not found"

**Cause**: Wrong path

**Solution**: Use absolute path or verify relative path
```bash
python main.py --data-path /full/path/to/reviews.json \
    --text-column text --label-column rating
```

### Poor Performance

**Cause**: Data quality issues

**Solution**: Enable cleaning
```bash
--cleaning all
```

### Out of Memory

**Cause**: Dataset too large or vector size too big

**Solution**: Reduce parameters
```bash
--vector-size 100 --min-count 5 --n-estimators 100
```

## 🎓 How It Works

1. **Data Loading**: Automatically detects JSON vs JSONL format
2. **Data Cleaning**: Optional preprocessing (configurable)
3. **Feature Extraction**: Word2Vec converts text to vectors
4. **Model Training**: XGBoost learns from the vectors
5. **Evaluation**: Cross-validation + test set metrics
6. **Visualization**: Generate confusion matrix and charts

## 📊 Performance Tips

### For Better Accuracy
- Enable all cleaning: `--cleaning all`
- Increase vector size: `--vector-size 400`
- More trees: `--n-estimators 300`

### For Faster Training
- Reduce vector size: `--vector-size 100`
- Fewer trees: `--n-estimators 50`
- Increase min-count: `--min-count 5`

### For Large Datasets
- Use JSONL format (line-by-line processing)
- Reduce vector size: `--vector-size 100`
- Increase min-count: `--min-count 10`

## 🔍 Data Requirements

### Minimum Requirements

1. **Format**: JSON or JSONL
2. **Structure**: Each item must be an object/dict
3. **Required Fields**: 
   - One text field (your reviews/content)
   - One label field (ratings/sentiment/scores)

### Recommended

- At least 1000 samples for good results
- Balanced label distribution
- Text length: 10-500 words per review
- Clean UTF-8 encoding

### What Works

✅ Any column names - you specify them
✅ Numeric labels (1-5, 0-1, 1-10, etc.)
✅ Text labels ("positive", "negative", etc.)
✅ Multiple labels (multiclass classification)
✅ Binary labels (2 classes)
✅ Additional numeric features

### What Doesn't Work

❌ Image data
❌ Audio data
❌ Multi-label classification (one review, multiple labels)
❌ Sequence prediction

## 🌍 Language Support

The pipeline works with **any language** that:
- Uses space-separated words
- Can be tokenized by splitting on whitespace

Tested with:
- English
- Spanish
- French
- German
- Italian

For languages without spaces (Chinese, Japanese, etc.), you may need to pre-tokenize.

## 📝 Complete Example Workflow

```bash
# 1. Prepare your data (JSON or JSONL)
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py \
    --data-path my_reviews.json \
    --text-column review_text \
    --label-column rating \
    --cleaning all \
    --output-path results/experiment_1.png

# 4. Check results
# - Console output shows metrics
# - results/experiment_1.png shows visualizations
```

## 🤝 Contributing

This is a generic pipeline - feel free to:
- Add more cleaning options
- Support more file formats
- Add more models
- Improve documentation

## 📄 License

MIT License - Free to use, modify, and distribute

---

**Remember**: This pipeline is **dataset-agnostic**. As long as you have review text and labels in JSON/JSONL format, it will work!
