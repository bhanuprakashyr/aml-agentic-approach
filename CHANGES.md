# Changes Log

## Done
- ✅ Created project folder: `/Users/bhanuprakash/Documents/AML - Project/aml-agentic-approach`
- ✅ Created EDA notebook: `notebooks/01_eda.ipynb` (complete)
  - Data loading & analysis
  - Class distribution, features, graph structure
- ✅ Created & ran GNN baseline notebook: `notebooks/02_gcn_baseline.ipynb`
  - GraphSAGE model trained (36 epochs, early stopping)
  - Test Results: F1=0.41, AUC=0.83, Recall=69%
  - Model checkpoint saved: `baseline/checkpoints/graphsage_baseline.pt`
  - Embeddings saved: `baseline/checkpoints/node_embeddings.npz`
- ✅ Created & ran Case Memory notebook: `notebooks/03_case_memory.ipynb`
  - 1000 cases selected (376 illicit, 624 licit)
  - GNNExplainer integrated for feature importance explanations
  - FAISS index built for similarity search
  - Saved to: `case_memory/`
- ✅ Created & ran RAG Pipeline notebook: `notebooks/04_rag_pipeline.ipynb`
  - End-to-end inference pipeline working
  - ICL prompt generation (risk-only, recommendation, full analysis)
  - Retrieval quality: Licit→Licit 99.6%, Illicit→Illicit 90.4%
  - Sample prompt saved to: `case_memory/sample_prompt_132832.txt`
  - **LLM Integration Complete:**
    - Unified LLMClient supporting OpenAI, Anthropic, Ollama
    - Tested with Ollama (llama3.2) - working ✅
    - Full fraud analysis with ICL prompts
- ✅ Environment setup:
  - Created `.env` file for API keys (OpenAI, Anthropic)
  - Added `.env` to `.gitignore` for security
  - Installed: `python-dotenv`, `openai` packages

## Recent Updates (Feb 2026)
- Improved retrieval quality significantly (Illicit→Illicit: 20% → 90.4%)
- LLM integration tested with Ollama llama3.2 (free, local)
- ICL prompts now include:
  - Transaction details (ID, fraud score, risk level)
  - Network context (neighbors, in/out degree)
  - Similar historical cases with verdicts
  - GNNExplainer insights from case memory
- Full analysis output includes risk assessment, pattern analysis, reasoning, and recommendations

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
