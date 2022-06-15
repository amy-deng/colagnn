# Colagnn

This is the source code for paper [Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction](https://yue-ning.github.io/docs/CIKM20-colagnn.pdf) appeared in CIKM2020 (research track)


## Raw Data
The raw dataset are in in the `data` folder. For each dataset, there are two files defined. For example, for the `Japan-prefecture` dataset, we have two files:
- `japan.txt` includes the spatiotemporal data. Columns indicate locations (i.e., prefecture) and rows indicate timestamps (i.e., weeks). Each value is the number of patients in a location at a time point. The data are arranged in chronological order.
- `japan-adj.txt` contains a adjacency matrix.


## Training Data
The training data are processed by the **DataBasicLoader**
class in the `src/data.py` file. We can set different value for historical window size **args.window** and horizon/leadtime **args.horizon**. Setting **args.window=20, args.horizon=1/2** means using data from the previous 20 weeks to predict the *upcoming*/*next* week. There are some functions in this class:
- **_split** splits the data into training/validation/test sets.
- **_batchify** generates data samples. Each sample contains a time series input with length equal to **args.window**, and a value for the output. For the current code, there are overlaps in the inputs of different samples.
- **get_batches** generates random mini-batches for training.
