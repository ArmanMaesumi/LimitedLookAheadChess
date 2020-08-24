# Playing Chess with Limited Look Ahead
Authors: Arman Maesumi

[(preprint)](https://arxiv.org/abs/2007.02130)

## Abstract
We have seen numerous machine learning methods tackle the game of chess over the years. However, one common element in these works is the necessity of a finely optimized look ahead algorithm. The particular interest of this research lies with creating a chess engine that is highly capable, but restricted in its look ahead depth. We train a deep neural network to serve as a static evaluation function, which is accompanied by a relatively simple look ahead algorithm. We show that our static evaluation function has encoded some semblance of look ahead knowledge, and is comparable to classical evaluation functions. The strength of our chess engine is assessed by comparing its proposed moves against those proposed by Stockfish. We show that, despite strict restrictions on look ahead depth, our engine recommends moves of equal strength in roughly 83% of our sample positions.

## Dependencies
Python dependencies for training and evaluating models (use pip install):

```
numpy==1.16.1
Keras==2.3.1
python-chess==0.30.1
tensorboard==1.15.0
tensorflow==1.15.0
```

Golang dependencies for executing look ahead algorithm (use go get):
```
golang.org/x/sync/syncmap
github.com/ArmanMaesumi/chess
github.com/galeone/tfgo
github.com/tensorflow/tensorflow/tensorflow/go
```

## Dataset Generation
To create the training and testing data, see `stockfish_eval.py`. This is a multi-threaded python script that uses Stockfish to label static positions. You will need to provide a `.pgn` file for the collection of board positions. We suggest using a `.pgn` from the [Lichess Database](https://database.lichess.org/), as this provides virtually limitless positions from human games. To perform artificial dataset expansion (as mentioned in the paper), you may use `expand_dataset.py`. This script will generate and label random positions, which we found to increase playing performance in the final model.

## Inference
A trained model can be downloaded here: https://drive.google.com/file/d/1Jns6TtHKrsNMj7Cx3burZxQHA3p2joIP/view?usp=sharing (~60 mb). Please extract the contents to `.\look_ahead\static_evaluation_model\`. This directory should now contain:

`.\look_ahead\static_evaluation_model\variables\...`, and `.\look_ahead\static_evaluation_model\saved_model.pb`

To run the look ahead algorithm in terminal (GPU required):
```
> cd .\look_ahead\
> go run blundr.go evaluation.go search.go
```
You will be prompted to input a chess position. Copy-paste a FEN string into the terminal to execute the look ahead algorithm. The program will iteratively output suggested moves starting at depth=2 to depth=5.

## Datasets

Due to the size of the datasets used in this paper, we cannot include them in this repository. The datasets will be made available by email request (arman@cs.utexas.edu). The primary dataset, which contains upwards of 30 million chess positions (21 million of which are labelled), is nearly 2 gigabytes. This dataset was used for training and testing the deep autoencoder, classifier, and static evaluation function. 
