# Changes Log

## Done
- ✅ Created project folder: `/Users/bhanuprakash/Documents/AML - Project/aml-agentic-approach`
- ✅ Created EDA notebook: `notebooks/01_eda.ipynb` (complete)
  - Data loading & analysis
  - Class distribution, features, graph structure

## Next
- [ ] Create GCN baseline notebook: `notebooks/02_gcn_baseline.ipynb`
  - Load data from EDA
  - Build GCN model
  - Train & evaluate

## Key Context
- **Dataset**: Elliptic (200K nodes, 234K edges, 166 features)
- **Location**: `/Users/bhanuprakash/Documents/AML - Project/aml-agentic-approach/`
- **Data path**: Download from https://www.kaggle.com/datasets/ellipticbenin/elliptic-data-set and put in `data/` folder
- **Strategy**: CPU only for dev, Colab GPU for training
- **Focus**: GCN → GAT → GraphSAGE models
