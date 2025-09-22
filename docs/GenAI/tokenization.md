## Tokenization in NLP

- **Definition**: The process of splitting text into smaller pieces (tokens) for model input.  
- **Types of Tokenization**:  
    - **Word-based**: Each word is a token. Preserves meaning but increases vocabulary size.  
    - **Character-based**: Splits into characters. Small vocabulary but less semantic meaning.  
    - **Subword-based**: Keeps common words whole, splits rare words into subwords. Balances vocabulary size and meaning.  

- **Popular Tokenizers**:  
    - **WordPiece** (used in BERT) → splits words based on frequency and meaning.  
    - **Unigram / SentencePiece** (used in XLNet) → breaks text into subwords, assigning IDs.

- **Special symbols**
    1. `<unk>`: This token represents "unknown" or "out-of-vocabulary" words. It is used when a word in the input text is not found in the vocabulary or when dealing with words that are rare or unseen during training. When the model encounters an unknown word, it replaces it with the `<unk>` token.
    2. `<pad>`: This token represents padding. In sequences of text data, such as sentences or documents, sequences may have different lengths. To create batches of data with uniform dimensions, shorter sequences are often padded with this `<pad>` token to match the length of the longest sequence in the batch.
    3. `<bos>`: This token represents the "beginning of sequence." It is used to indicate the start of a sentence or sequence of tokens. It helps the model understand the beginning of a text sequence.
    4. `<eos>`: This token represents the "end of sequence." It is used to indicate the end of a sentence or sequence of tokens. It helps the model recognize the end of a text sequence.
    5. `MASK` (Masked Token): Utilized for word replacement in tasks like masked language modeling. It allows models to predict the identity of masked-out words, facilitating learning of bidirectional representations.

- **Practical Notes**:  
    - Tools like **NLTK**, **spaCy**, and **torchtext** support tokenization.  
    - Special tokens such as **[BOS]** (beginning of sentence) and **[EOS]** (end of sentence) help models understand sequence boundaries.  
    - Padding tokens ensure uniform input lengths for training.  


---