# Changes Log

## Done
- ✅ Created project folder: `/Users/bhanuprakash/Documents/AML - Project/aml-agentic-approach`
- ✅ Created EDA notebook: `notebooks/01_eda.ipynb` (complete)
  - Data loading & analysis
  - Class distribution, features, graph structure
- ✅ Created & ran GNN baseline notebook: `notebooks/02_gcn_baseline.ipynb`
  - GraphSAGE model trained (36 epochs, early stopping)
  - Test Results: F1=0.15, AUC=0.77, Recall=69%
  - Model checkpoint saved: `baseline/checkpoints/graphsage_baseline.pt`
  - Embeddings saved: `baseline/checkpoints/node_embeddings.npz`

## Next
- [ ] Build Case Memory: `notebooks/03_case_memory.ipynb`
  - Load embeddings from baseline
  - Create FAISS index
  - Build case store with explanations
- [ ] Test RAG Pipeline: `notebooks/04_rag_pipeline.ipynb`
  - End-to-end inference
  - ICL prompt generation
  - LLM integration

## Key Context
- **Dataset**: Elliptic (200K nodes, 234K edges, 166 features)
- **Location**: `/Users/bhanuprakash/Documents/AML - Project/aml-agentic-approach/`
- **Data path**: Download from https://www.kaggle.com/datasets/ellipticbenin/elliptic-data-set and put in `data/` folder
- **Strategy**: CPU only for dev, Colab GPU for training
- **Focus**: GCN → GAT → GraphSAGE models
