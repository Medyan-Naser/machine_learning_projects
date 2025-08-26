## Tokenization in NLP

- **Definition**: The process of splitting text into smaller pieces (tokens) for model input.  
- **Types of Tokenization**:  
    - **Word-based**: Each word is a token. Preserves meaning but increases vocabulary size.  
    - **Character-based**: Splits into characters. Small vocabulary but less semantic meaning.  
    - **Subword-based**: Keeps common words whole, splits rare words into subwords. Balances vocabulary size and meaning.  

- **Popular Tokenizers**:  
    - **WordPiece** (used in BERT) → splits words based on frequency and meaning.  
    - **Unigram / SentencePiece** (used in XLNet) → breaks text into subwords, assigning IDs.  

- **Practical Notes**:  
    - Tools like **NLTK**, **spaCy**, and **torchtext** support tokenization.  
    - Special tokens such as **[BOS]** (beginning of sentence) and **[EOS]** (end of sentence) help models understand sequence boundaries.  
    - Padding tokens ensure uniform input lengths for training.  


---