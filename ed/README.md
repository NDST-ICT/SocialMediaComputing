# Encoder-Decoder

This repository is a re-implementation of Encoder-Decoder community detection model proposed in the following paper:

> Sun, Bing-Jie, et al. "A Non-negative Symmetric Encoder-Decoder Approach for Community Detection." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. ACM, 2017.

## Introduction

The Encoder-Decoder model is a kind of non-negative matrix factorization. The original input is represented as a matrix A, where each row corresponds to a feature and each column corresponds to a sample or an instance over these features. As for the task of community detection, A is always the adjacency matrix. This model aims to factorize the original input matrix A into a basis matrix W and a code matrix Z, which are both non-negative. The size of A, W, Z is n × m, n × k, k × m respectively. It has an encoder and a decoder.

An encoder requires that Z ≈ W^TA and a decoder constrains A ≈ WZ, resulting in A ≈ WW^TA. Thus the basis matrix W is non-negative and sparse, which makes it well reveal the community structure of original input data.

## Example

We adopt the karate network as an example to illustrate the model.

The input data is an adjacency matrix of the network:

    [[0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 0. 0. 0. 1. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [1. 0. 1. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [1. 1. 0. 1. 0. 0. 0. 1. 1. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 1. 0.]
     [1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 1.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1.]
     [0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1.]
     [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1.]
     [1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 1. 0. 0. 1. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 1. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 1.]
     [0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 1. 0. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 1. 1. 1. 0. 0. 1. 1. 1. 0. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 0.]]

In the example, we set k to 2. After factorization, we get a basis matrix W, which can be seen as the k-dimension representations for the nodes in karate network:

	[[5.25601674e-001 1.10549012e-017]
	 [3.81709448e-001 6.44314960e-016]
	 [3.22259164e-001 1.01795252e-001]
	 [3.24039986e-001 1.36527319e-023]
	 [1.37364529e-001 5.61983360e-047]
	 [1.46130384e-001 1.60467925e-049]
	 [1.46130384e-001 4.75195694e-049]
	 [2.63828817e-001 1.47159409e-021]
	 [1.09291309e-001 1.97790377e-001]
	 [3.24363678e-002 1.04880488e-001]
	 [1.37364529e-001 1.16977497e-046]
	 [8.72783620e-002 9.09494948e-033]
	 [1.44056872e-001 1.03442021e-051]
	 [2.60578949e-001 4.52570560e-002]
	 [4.41793435e-046 1.61296729e-001]
	 [2.25844348e-046 1.61296729e-001]
	 [5.16349921e-002 1.06649152e-117]
	 [1.52246245e-001 9.02956688e-036]
	 [2.83764721e-045 1.61296729e-001]
	 [1.45745338e-001 5.54033241e-002]
	 [2.45997727e-046 1.61296729e-001]
	 [1.52246245e-001 3.82668160e-036]
	 [4.41351659e-046 1.61296729e-001]
	 [2.85058753e-040 2.46070537e-001]
	 [2.80310038e-009 8.16798067e-002]
	 [1.15655973e-027 9.65076749e-002]
	 [2.76663219e-058 1.28228809e-001]
	 [5.16207501e-007 1.74057497e-001]
	 [3.33053030e-002 1.42281249e-001]
	 [8.15829791e-049 2.28611225e-001]
	 [4.76054555e-002 1.86857921e-001]
	 [4.66853007e-002 2.10535144e-001]
	 [2.32745374e-009 4.56102765e-001]
	 [3.18298277e-005 5.35734103e-001]]

For each node, we choose the column index with a larger value in its representation to be its community label. 

Finally, we get the result of community division in karate network:

	[0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1]

## DataSet

In order to test this model, we adopt the “benchmark” tool to generate synthetic networks which is introduced in:

> Lancichinetti, Andrea, and Santo Fortunato. "Benchmarks for testing community detection algorithms on directed and weighted graphs with overlapping communities." Physical Review E 80.1 (2009): 016118.

 And we download Amazon and DBLP real-world network datasets from  [http://snap.stanford.edu/data/](http://snap.stanford.edu/data/).

The “benchmark” tool is provided in "benchmark.zip".

## Steps to run the model

1.generate synthetic network or download real-world network from SNAP

2.convert network data to the form which the model can recognize

command:

    python3 converter_benchmark.py -net network_file_path -com community_file_path
or:

    python3 converter_SNAP.py -net network_file_path -label label_file_path

2.run the model to detect community

command:

    python3 Encoder-Decoder_scipy.py (Encoder-Decoder_tf.py, NMF_scipy.py, NMF_sklearn.py) -k number_of_communities

4.show the result of NMI for covers (on synthetic network) or purity (on real-world network)

command:

    ./mutual file1.txt file2.txt
	
or:

    python3 show_result.py -c purity -k number_of_communities
	
The tool to calculate NMI for covers is provided in "mutual3.tar.gz", and it is proposed in this paper:

> Lancichinetti, Andrea, S. Fortunato, and J. Kertész. "Detecting the overlapping and hierarchical community structure of complex networks." New Journal of Physics 11.3(2008):19-44.

## My results

On synthetic networks of different sizes, the results of NMI for covers are shown in the following tables:

| model | NMF | Encoder-Decoder |
| :------------: | :------------: | :------------: |
| 500_nodes_1th | 0.817 | 0.830 |
| 500_nodes_2th | 0.860 | 0.921 |
| 500_nodes_3th | 0.834 | 0.843 |
| 500_nodes_4th | 0.848 | 0.830 |
| 500_nodes_5th | 0.888 | 0.935 |
| 500_nodes_6th | 0.920 | 0.848 |
| 500_nodes_7th | 0.883 | 0.955 |
| 500_nodes_8th | 0.900 | 0.898 |
| 500_nodes_9th | 0.835 | 0.88 |
| 500_nodes_10th | 0.841 | 0.829 |
| 500_nodes_maximum | 0.920 | 0.955 |

| model | NMF | Encoder-Decoder |
| :------------: | :------------: | :------------: |
| 5000_nodes_1th | 0.962 | 0.944 |
| 5000_nodes_2th | 1.000 | 0.963 |
| 5000_nodes_3th | 1.000 | 1.000 |
| 5000_nodes_4th | 1.000 | 1.000 |
| 5000_nodes_5th | 0.998 | 1.000 |
| 5000_nodes_6th | 1.000 | 0.956 |
| 5000_nodes_7th | 1.000 | 0.957 |
| 5000_nodes_8th | 1.000 | 1.000 |
| 5000_nodes_9th | 1.000 | 0.962 |
| 5000_nodes_10th | 1.000 | 0.959 |
| 5000_nodes_maximum | 1.000 | 1.000 |


| model | NMF | Encoder-Decoder |
| :------------: | :------------: | :------------: |
| 10000_nodes_1th | 1.000 | 1.000 |
| 10000_nodes_2th | 1.000 | 1.000 |
| 10000_nodes_3th | 1.000 | 0.999 |
| 10000_nodes_4th | 0.963 | 0.962 |
| 10000_nodes_5th | 1.000 | 1.000 |
| 10000_nodes_6th | 1.000 | 1.000 |
| 10000_nodes_7th | 1.000 | 0.964 |
| 10000_nodes_8th | 1.000 | 1.000 |
| 10000_nodes_9th | 1.000 | 1.000 |
| 10000_nodes_10th | 1.000 | 1.000 |
| 10000_nodes_maximum | 1.000 | 1.000 |

On real-world networks, these are the results of purity:

| model | NMF | Encoder-Decoder |
| :------------: | :------------: | :------------: |
| DBLP_1th | 0.167 | 0.147 |
| DBLP_2th | 0.166 | 0.129 |
| DBLP_3th | 0.155 | 0.146 |
| DBLP_4th | 0.164 | 0.147 |
| DBLP_5th | 0.161 | 0.149 |
| DBLP_6th | 0.164 | 0.158 |
| DBLP_7th | 0.161 | 0.130 |
| DBLP_8th | 0.160 | 0.146 |
| DBLP_9th | 0.167 | 0.147 |
| DBLP_10th | 0.154 | 0.136 |
| DBLP_maximum | 0.167 | 0.158 |

| model | NMF | Encoder-Decoder |
| :------------: | :------------: | :------------: |
| Amazon_1th | 0.415 | 0.406 |
| Amazon_2th | 0.439 | 0.363 |
| Amazon_3th | 0.375 | 0.427 |
| Amazon_4th | 0.414 | 0.388 |
| Amazon_5th | 0.377 | 0.411 |
| Amazon_6th | 0.391 | 0.392 |
| Amazon_7th | 0.379 | 0.375 |
| Amazon_8th | 0.404 | 0.374 |
| Amazon_9th | 0.398 | 0.381 |
| Amazon_10th | 0.402 | 0.431 |
| Amazon_maximum | 0.439 | 0.431 |