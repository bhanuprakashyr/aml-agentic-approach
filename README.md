# Agentic AML System

An intelligent Anti-Money Laundering (AML) system that combines Graph Neural Networks (GNN) with Retrieval-Augmented Generation (RAG) for explainable fraud detection on blockchain transactions.

## 🎯 Overview

This project implements a novel approach to AML by:
1. **GNN-based Detection**: Using GraphSAGE/GAT models to detect suspicious transactions in the Bitcoin network
2. **Case Memory**: Storing historical fraud cases with explanations
3. **RAG Pipeline**: Retrieving similar past cases to provide context-aware, explainable predictions
4. **In-Context Learning**: Generating human-readable explanations using retrieved cases as examples

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input Transaction                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Graph Neural Network                            │
│            (GraphSAGE / GAT Encoder)                            │
│                                                                 │
│  • Node feature extraction                                      │
│  • Neighborhood aggregation                                     │
│  • Transaction embedding generation                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RAG Pipeline                                  │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ FAISS Index  │───▶│  Retriever   │───▶│ ICL Builder  │      │
│  │              │    │              │    │              │      │
│  │ Case vectors │    │ Top-k cases  │    │ Prompt       │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Explainable Prediction                             │
│                                                                 │
│  • Fraud probability score                                      │
│  • Similar historical cases                                     │
│  • Human-readable explanation                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
aml-agentic-approach/
├── configs/
│   ├── model.yaml          # GNN model configuration
│   ├── training.yaml       # Training hyperparameters
│   └── rag.yaml            # RAG pipeline settings
├── data/
│   └── elliptic_bitcoin_dataset/   # (not tracked in git)
│       ├── elliptic_txs_features.csv
│       ├── elliptic_txs_edgelist.csv
│       └── elliptic_txs_classes.csv
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── src/
│   ├── data/
│   │   ├── elliptic_loader.py    # Dataset loading utilities
│   │   └── graph_builder.py      # PyG graph construction
│   ├── models/
│   │   ├── graphsage.py          # GraphSAGE implementation
│   │   └── gat.py                # Graph Attention Network
│   ├── explainer/
│   │   └── gnn_explainer.py      # GNN explanation generation
│   ├── memory/
│   │   ├── case_store.py         # Historical case storage
│   │   └── case_selector.py      # Case selection strategies
│   ├── retrieval/
│   │   ├── faiss_index.py        # FAISS vector index
│   │   └── retriever.py          # Similar case retrieval
│   ├── prompts/
│   │   ├── icl_constructor.py    # In-context learning prompts
│   │   └── templates.py          # Prompt templates
│   ├── pipeline/
│   │   └── inference.py          # End-to-end inference
│   ├── utils/
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── visualization.py      # Plotting utilities
│   └── train.py                  # Training script
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.9+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/bhanuprakashyr/aml-agentic-approach.git
cd aml-agentic-approach

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- PyTorch
- PyTorch Geometric
- FAISS
- NumPy, Pandas, Scikit-learn
- PyYAML
- Matplotlib, Seaborn

## 📊 Dataset

This project uses the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), which contains:
- **203,769** Bitcoin transactions
- **234,355** directed payment flows (edges)
- **166** node features (timestamps + transaction features)
- Labels: **illicit** (4,545), **licit** (42,019), **unknown** (157,205)

### Download Data
1. Download from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
2. Extract to `data/elliptic_bitcoin_dataset/`

## 🎮 Usage

### Training the GNN Model

```python
from src.train import train_model
from src.data.elliptic_loader import EllipticDataset
from src.data.graph_builder import build_pyg_graph

# Load and prepare data
dataset = EllipticDataset("data/elliptic_bitcoin_dataset")
graph = build_pyg_graph(dataset)

# Train model
model = train_model(graph, config_path="configs/training.yaml")
```

### Running Inference

```python
from src.pipeline.inference import AMLInferencePipeline

# Initialize pipeline
pipeline = AMLInferencePipeline(
    model_path="checkpoints/best_model.pt",
    config_path="configs/rag.yaml"
)

# Get prediction with explanation
result = pipeline.predict(transaction_id=12345)

print(f"Fraud Probability: {result['probability']:.2%}")
print(f"Explanation: {result['explanation']}")
print(f"Similar Cases: {result['similar_cases']}")
```

## 🧠 Model Details

### GraphSAGE
- **Aggregator**: Mean/LSTM/Pool
- **Hidden dimensions**: 128
- **Number of layers**: 2
- **Dropout**: 0.5

### Graph Attention Network (GAT)
- **Attention heads**: 8
- **Hidden dimensions**: 128
- **Number of layers**: 2
- **Dropout**: 0.6

## 📈 Evaluation Metrics

- **Precision / Recall / F1-Score**
- **AUC-ROC**
- **Average Precision (AP)**
- **Illicit F1** (primary metric for imbalanced data)

## 🔮 Future Work

- [ ] Integration with LLM APIs for enhanced explanations
- [ ] Real-time transaction monitoring
- [ ] Multi-chain support (Ethereum, etc.)
- [ ] Active learning for continuous improvement
- [ ] Dashboard for visualization

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- [Elliptic](https://www.elliptic.co/) for the Bitcoin dataset
- PyTorch Geometric team
- FAISS by Meta AI
