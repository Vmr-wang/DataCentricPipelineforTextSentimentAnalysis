"""
Feature Extraction Module
Handles Word2Vec training and text-to-vector transformation
"""

import numpy as np
from gensim.models import Word2Vec


def tokenize_texts(texts):
    """
    Tokenize texts into list of words (simple whitespace tokenization)
    
    Args:
        texts: Series or list of text strings
    
    Returns:
        list: List of tokenized sentences
    """
    return [text.lower().split() for text in texts]


def train_word2vec(X_train, vector_size=300, window=5, min_count=2, workers=4):
    """
    Train Word2Vec model on training data
    
    Args:
        X_train: Training text data
        vector_size (int): Dimension of word vectors
        window (int): Context window size
        min_count (int): Minimum word frequency
        workers (int): Number of worker threads
    
    Returns:
        Word2Vec: Trained Word2Vec model
    """
    print("\n" + "=" * 60)
    print("STEP 3: WORD2VEC FEATURE EXTRACTION")
    print("=" * 60)
    
    print("Tokenizing training texts...")
    tokenized_train = tokenize_texts(X_train)
    print(f"✓ Tokenized {len(tokenized_train)} documents")
    
    # Calculate total words
    total_words = sum(len(sent) for sent in tokenized_train)
    avg_words = total_words / len(tokenized_train)
    print(f"  Total words: {total_words:,}")
    print(f"  Average words per document: {avg_words:.2f}")
    
    print(f"\nTraining Word2Vec model...")
    print(f"  Parameters:")
    print(f"    - Vector size: {vector_size}")
    print(f"    - Window size: {window}")
    print(f"    - Min count: {min_count}")
    print(f"    - Algorithm: Skip-gram")
    
    w2v_model = Word2Vec(
        sentences=tokenized_train,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # 1 = Skip-gram, 0 = CBOW
        seed=42
    )
    
    print(f"\n✓ Word2Vec model trained successfully")
    print(f"  Vocabulary size: {len(w2v_model.wv)} words")
    print(f"  Vector dimension: {vector_size}")
    
    # Show some example word vectors
    print(f"\nSample words in vocabulary:")
    sample_words = list(w2v_model.wv.index_to_key[:10])
    print(f"  {sample_words}")
    
    return w2v_model


def text_to_vector(text, w2v_model, vector_size=300):
    """
    Convert a single text to vector by averaging word vectors
    
    Args:
        text (str): Input text
        w2v_model: Trained Word2Vec model
        vector_size (int): Dimension of vectors
    
    Returns:
        np.array: Document vector
    """
    words = text.lower().split()
    word_vectors = []
    
    for word in words:
        if word in w2v_model.wv:
            word_vectors.append(w2v_model.wv[word])
    
    # If no words found in vocabulary, return zero vector
    if len(word_vectors) == 0:
        return np.zeros(vector_size)
    
    # Return mean of all word vectors
    return np.mean(word_vectors, axis=0)


def transform_texts_to_vectors(texts, w2v_model, vector_size=300):
    """
    Transform multiple texts to vectors
    
    Args:
        texts: Series or list of texts
        w2v_model: Trained Word2Vec model
        vector_size (int): Dimension of vectors
    
    Returns:
        np.array: Matrix of document vectors
    """
    print(f"\nTransforming texts to vectors...")
    vectors = []
    
    for i, text in enumerate(texts):
        vectors.append(text_to_vector(text, w2v_model, vector_size))
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(texts)} documents...")
    
    vectors = np.array(vectors)
    print(f"✓ Transformation completed")
    print(f"  Output shape: {vectors.shape}")
    
    return vectors
