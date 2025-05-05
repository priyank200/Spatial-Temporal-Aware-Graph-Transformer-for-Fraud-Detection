# Spatial-Temporal-Aware-Graph-Transformer-for-Fraud-Detection

## Overview

This project implements a Spatio-Temporal Graph Attention Network (STGAT) model for analyzing transaction data represented as a heterogeneous graph, aimed at detecting money laundering in financial transactions. The implementation includes two main components:

1. **Graph Creation**: A script to preprocess transaction data and construct a heterogeneous graph using PyTorch Geometric.
2. **Model Training**: A script to train the STGAT model on the constructed graph for transaction classification.

The datasets are sourced from Kaggle, with both subset and complete versions available for experimentation.

## Datasets

The project uses the following Kaggle datasets:

- **Subset Dataset**: A smaller dataset (157,968 transactions) for quick experimentation.
  - Link: [Small Dataset](https://www.kaggle.com/datasets/priyankmundra/small-dataset-1)
- **Subset Graphical Dataset**: Preprocessed graph data for the subset dataset.
  - Link: [Graph Dataset](https://www.kaggle.com/datasets/priyankmundra/graph-dataset)
- **Complete Dataset**: The full dataset (9,504,852 transactions, 0.1% fraudulent) for comprehensive analysis.
  - Link: [Complete Dataset](https://www.kaggle.com/datasets/priyankmundra/complete-dataset)
- **Complete Graphical Dataset**: Preprocessed graph data for the complete dataset.
  - Link: [Complete Graph](https://www.kaggle.com/datasets/priyankmundra/complete-graph)

### Dataset Description

The datasets contain transaction records with the following key features:
- `Sender_account`, `Receiver_account`: Account identifiers.
- `Sender_bank_location`, `Receiver_bank_location`: Bank locations (encoded as integers).
- `Payment_currency`, `Received_currency`: Currencies used (encoded as integers).
- `Amount`: Transaction amount.
- `Payment_type`: Type of payment (encoded as integer).
- `Year`, `Month`, `Day`: Transaction date.
- `Is_laundering`: Binary label for money laundering (1 for laundering, 0 otherwise).
- `Laundering_type`: Type of laundering (if applicable).

The graph is heterogeneous, with two node types (`user` and `transaction`) and two edge types (`user-sends-transaction` and `transaction-received_by-user`).

## Prerequisites

Ensure the following dependencies are installed:

- Python 3.10 or higher
- PyTorch (version 2.5.1 with CUDA 12.1)
- PyTorch Geometric
- Additional libraries: `pandas`, `numpy`, `networkx`, `matplotlib`, `tqdm`, `scikit-learn`, `seaborn`

Install dependencies using the commands in `Graph_Creation.ipynb`:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install pandas numpy torch torch_geometric networkx matplotlib
```

## Project Structure

- **Graph_Creation.ipynb**: Notebook for creating the heterogeneous graph from transaction data.
  - Constructs a `HeteroData` object with user and transaction nodes.
  - Features include one-hot encoded payment types, currencies, locations, and transaction amounts.
  - Saves the graph to a `.pt` file.
- **Training.ipynb**: Notebook for training the STGAT model.
  - Implements the STGAT model using `HeteroConv`, temporal encoding, and Transformer layers.
  - Trains the model to classify transactions as laundering or non-laundering.
- **README.md**: This file, providing an overview and instructions.

## Graph Creation

The `Graph_Creation.ipynb` notebook performs the following:

1. **Data Loading**: Reads the transaction CSV with optimized data types.
2. **User Mapping**: Maps account numbers to user indices and computes the most frequent currency and location per user.
3. **Node Initialization**:
   - **Transaction Nodes**: Features include transaction amount, one-hot encoded payment type, and date.
   - **User Nodes**: Features include one-hot encoded most frequent currency and location.
4. **Edge Creation**: Adds `user-sends-transaction` and `transaction-received_by-user` edges.
5. **Graph Saving**: Saves the `HeteroData` object to a `.pt` file.

To run graph creation:

1. Update `csv_file` and `graph_path` in `HetroGraphDataset` to point to your dataset and output path.
2. Set `force_recreate=True` to recreate the graph or `False` to load an existing graph.
3. Execute the notebook cells.

Example:
```python
dataset = HetroGraphDataset(
    csv_file='/kaggle/input/complete-dataset/data.csv',
    graph_path='/kaggle/input/complete-graph/complete-graph.pt',
    force_recreate=True
)
dataset.info_about_graph()
```

## Model Training

The `Training.ipynb` notebook trains the STGAT model with the following steps:

1. **Graph Loading**: Loads the preprocessed graph from the `.pt` file.
2. **Model Definition**: Implements the STGAT model with:
   - **Temporal Encoding**: Embeds transaction timestamps using positional encoding.
   - **Heterogeneous Convolution**: Uses `SAGEConv` for relation-specific updates.
   - **Relation-Level Attention**: Weights edge types to prioritize relevant relationships.
   - **Transformer Encoder**: Captures sequential dependencies among transaction nodes.
   - **MLP Classification**: Outputs binary fraud predictions.
3. **Training Loop**:
   - Splits data chronologically (70% training, 30% testing) to maintain temporal consistency.
   - Uses weighted cross-entropy loss to address class imbalance (0.1% fraudulent transactions).
   - Employs `HGTLoader` for batch processing (batch size: 4096).
   - Optimizes with Adam (learning rate: 0.01).
4. **Evaluation**: Measures recall, accuracy, AUC, and harmonic mean (HM) score:
   - **Subset Dataset**:
     - Recall: 0.8381 (83.81% of fraudulent transactions detected)
     - Accuracy: 0.7614 (76.14%)
     - AUC: 0.8865
     - HM Score: 0.7979
   - **Complete Dataset**:
     - Recall: 0.7314 (73.14%)
     - Accuracy: 0.7791 (77.91%)
     - AUC: 0.8400
     - HM Score: 0.7545
5. **Visualization**: Generates confusion matrices, ROC curves, and loss plots.

To run training:

1. Ensure the graph file is available.
2. Update dataset paths in the notebook if necessary.
3. Execute the notebook cells.

## Usage

1. **Prepare the Environment**:
   - Install dependencies.
   - Download the desired dataset from Kaggle.
2. **Create the Graph**:
   - Run `Graph_Creation.ipynb` to generate the graph.
   - Verify the graph structure using `dataset.info_about_graph()`.
3. **Train the Model**:
   - Run `Training.ipynb` to train the STGAT model.
   - Monitor progress and evaluate results using generated plots.
4. **Experimentation**:
   - Use the subset dataset for quick prototyping.
   - Switch to the complete dataset for large-scale analysis.

## Results

The STGAT model was evaluated on both subset and complete datasets, achieving robust performance:

- **Subset Dataset (157,968 transactions)**:
  - High recall (83.81%) indicates effective fraud detection.
  - AUC of 0.8865 shows strong discriminative ability.
  - HM score of 0.7979 reflects balanced performance.
  - High false positives (10,800) suggest potential operational challenges.
- **Complete Dataset (9,504,852 transactions)**:
  - Recall of 73.14% demonstrates scalability.
  - AUC of 0.8400 and HM score of 0.7545 indicate robust performance on large-scale data.

Visualizations (confusion matrices, ROC curves, loss plots) are generated during training.

## Notes

- GPU access is recommended for faster training; CPU fallback is supported.
- Memory optimization in graph creation handles large datasets efficiently.
- The `force_recreate` flag allows reusing preprocessed graphs.
- Ensure dataset paths match your environment.
- The STGAT model uses a chronological train-test split to mimic real-world scenarios.

## Acknowledgments

- Graphical datasets provided by [Priyank Mundra](https://www.kaggle.com/priyankmundra) on Kaggle.
- Built using [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).
