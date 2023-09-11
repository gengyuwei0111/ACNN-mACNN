# ACNN-mACNN
# A Deep Learning Method for the Dynamics of Classic and Conservative Allen-Cahn Equations Based on Fully-Discrete Operators
Code repository for the paper:  
**A Deep Learning Method for the Dynamics of Classic and Conservative Allen-Cahn Equations Based on Fully-Discrete Operators**  
[Yuwei Geng](https://sc.edu/study/colleges_schools/artsandsciences/mathematics/our_people/directory/geng-yuwei.php), [Yuankai Teng](https://slooowtyk.github.io), [Zhu Wang](https://people.math.sc.edu/wangzhu) and [Lili Ju](https://people.math.sc.edu/ju)

<br>
Journal of Computational Physics, in revision 
<br>
<!--
[[paper](https://epubs.siam.org/doi/abs/10.1137/21M1459198)]


## Training Usage
To train the PRNN for a problem on given domain and draw a graph for regression
```shell
python ./train_model.py
 --case 2 
 --dim 2 
 --hidden_layers 2 
 --hidden_neurons 20 
 --lam_adf 1 
 --lam_bd 1 
 --optimizer 'Adam' 
 --Test_Mode 'LocalFitting' 
 --epochs_Adam 5000 
 --epochs_LBFGS 200 
 --TrainNum 2000 
 --coeff_para 50 
 --sigma 0.01 
 --domain 0 1
```

## Testing  Usage
To evaluate numerical error and relative sensitivity
```shell
python ./evaluate_model.py
 --case 2 
 --dim 2 
 --hidden_layers 2 
 --hidden_neurons 20 
 --lam_adf 1 
 --lam_bd 1 
 --optimizer 'Adam' 
 --Test_Mode 'LocalFitting' 
 --epochs_Adam 5000 
 --epochs_LBFGS 200 
 --TrainNum 2000 
 --coeff_para 50 
 --sigma 0.01 
 --domain 0 1
```


## Citation
If you  find the idea or code of this paper useful for your research, please consider citing us:

```bibtex
@article{teng2023level,
  title={Level Set Learning with Pseudoreversible Neural Networks for Nonlinear Dimension Reduction in Function Approximation},
  author={Teng, Yuankai and Wang, Zhu and Ju, Lili and Gruber, Anthony and Zhang, Guannan},
  journal={SIAM Journal on Scientific Computing},
  volume={45},
  number={3},
  pages={A1148--A1171},
  year={2023},
  publisher={SIAM}
}
```
-->
