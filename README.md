# DSEE

An example:
```shell
python3 main.py --metric L1 --loss mse --model M_GRU --rescale False
```
which means: 
+ using L1 distance between the embedding vectors as the approximated distance; 
+ using mean squared error on the approximated and groundtruth distances as the optimizing target;
+ using the GRU model as the embeding function;
+ do not further rescale the embedding vector, which is normed by the BN layer, to make the expected value of the approximated distance to meet the statistical average on the non-homologous sequences. 
----------
If this code is helpful, you may want to cite our work
```

@InProceedings{pmlr-v162-guo22f,
  title = 	 {Deep Squared {E}uclidean Approximation to the Levenshtein Distance for {DNA} Storage},
  author =       {Guo, Alan J.X. and Liang, Cong and Hou, Qing-Hu},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {8095--8108},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
}

```
------------
## DATA
You can make your own data from the DNA sequencing data which is publicly available from
```
@article{
doi:10.1126/science.aaj2038,
author = {Yaniv Erlich and Dina Zielinski },
title = {DNA Fountain enables a robust and efficient storage architecture},
journal = {Science},
volume = {355},
number = {6328},
pages = {950-954},
year = {2017},
doi = {10.1126/science.aaj2038},
}
```
The DSEE code uses data similar with 'dummy_data.npz',
```python
import numpy as np
data = np.load('dummy_data.npz')
data_unbatch = data['data_unbatch']
y_unbatch = data['y_unbatch']
print(data_unbatch.shape)
print(y_unbatch.shape)
```
```
(1000, 2, 5, 160)
(1000)
```
where the `1000` is the total number of samples, each sample is a pair of `2` DNA sequences, each base in the DNA sequences is represented in the one-hot form of `5` letters `ATGCN`, and the `160` is the padded length of the DNA sequences. `y` is the ground truth Levenshtein distance. 
