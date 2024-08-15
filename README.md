# 🤖 Detect AI Generated Text 📜

This project aims to detect AI-generated text using a combination of techniques, including BERT embeddings, linguistic features, and vector similarity. The goal is to accurately identify text that has been generated by AI models, such as GPT-3, and distinguish it from human-written content.

## 🌟 Features

- **BERT Embeddings**: Utilizes the power of BERT to generate embeddings for text, capturing semantic and contextual information.
- **Linguistic Features**: Analyzes various linguistic aspects of the text, including:
  - Flesch Reading Ease 📖
  - Flesch-Kincaid Grade Level 🎓
  - Gunning Fog Index 🌫️
  - Word Frequencies and Distributions 📊
- **Pinecone Vector DB**: Leverages Pinecone as a vector database to efficiently store and retrieve embeddings.
- **Cosine Similarity**: Computes the cosine similarity between text embeddings to measure their similarity.
- **Perplexity and Burstiness Score**: Utilizes OpenAI's API to calculate perplexity and burstiness scores, providing additional insights into the text's characteristics.
- **Streamlit App**: Provides a user-friendly interface through a Streamlit app, allowing users to input text and receive AI generation detection results.

## 📚 Methodology

The project combines multiple approaches to detect AI-generated text:

1. **BERT Embeddings**: The text is transformed into high-dimensional vectors using BERT, capturing its semantic and contextual information.

2. **Linguistic Features**: Various linguistic features, such as readability scores and word distributions, are extracted from the text to identify patterns and characteristics associated with AI-generated content.

3. **Vector Similarity**: The cosine similarity between the text embeddings is computed to measure the similarity between the input text and known AI-generated text samples.

4. **Perplexity and Burstiness Score**: OpenAI's API is utilized to calculate the perplexity and burstiness scores of the text, providing additional indicators of AI-generated content.

5. **Classification**: The extracted features and similarity scores are used to train a classifier that determines whether the input text is likely to be AI-generated or human-written.

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. Let's collaborate to improve the AI-generated text detection capabilities of this project.
