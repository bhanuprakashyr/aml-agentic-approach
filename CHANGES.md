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
- ✅ Created & ran Case Memory notebook: `notebooks/03_case_memory.ipynb`
  - 1000 cases selected (376 illicit, 624 licit)
  - FAISS index built for similarity search
  - Saved to: `case_memory/`
- ✅ Created & ran RAG Pipeline notebook: `notebooks/04_rag_pipeline.ipynb`
  - End-to-end inference pipeline working
  - ICL prompt generation (risk-only, recommendation, full analysis)
  - Retrieval quality: Licit→Licit 94%, Illicit→Illicit 20%
  - Sample prompt saved to: `case_memory/sample_prompt_160387.txt`
  - Ready for LLM integration (GPT-4, Claude, etc.)

## Next
- [ ] Improve model F1 score (currently 0.15)
- [ ] Add GNNExplainer for feature importance
- [ ] Integrate with production LLM
- [ ] Build analyst dashboard

## Key Context
- **Dataset**: Elliptic (200K nodes, 234K edges, 166 features)
- **Location**: `/Users/bhanuprakash/Documents/AML - Project/aml-agentic-approach/`
- **Data path**: Download from https://www.kaggle.com/datasets/ellipticbenin/elliptic-data-set and put in `data/` folder
- **Strategy**: CPU only for dev, Colab GPU for training
- **Focus**: GCN → GAT → GraphSAGE models
