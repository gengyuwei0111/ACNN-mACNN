# ACNN-mACNN
# A Deep Learning Method for the Dynamics of Classic and Conservative Allen-Cahn Equations Based on Fully-Discrete Operators
Code repository for the paper:  
**A Deep Learning Method for the Dynamics of Classic and Conservative Allen-Cahn Equations Based on Fully-Discrete Operators**  
[Yuwei Geng](https://sc.edu/study/colleges_schools/artsandsciences/mathematics/our_people/directory/geng-yuwei.php), [Yuankai Teng](https://slooowtyk.github.io), [Zhu Wang](https://people.math.sc.edu/wangzhu) and [Lili Ju](https://people.math.sc.edu/ju)
<br>
Journal of Computational Physics, to appear. 
<br>

<!--
[[paper](https://epubs.siam.org/doi/abs/10.1137/21M1459198)]
-->
## Training Usage

```shell
python ./train_model.py --height 128 --width 128 --intial_snap 1 --mid_channels 16 --deltaT 0.1 --trainingBatch 1 --testBatch 1 --epsilon 0.01
```

## Testing  Usage

```shell
python ./tester.py --TestingEndingTime 10 --height 128 --width 128 --intial_snap 1 --mid_channels 16 --deltaT 0.1 --trainingBatch 1 --testBatch 1 --epsilon 0.01
```



## Citation
If you  find the idea or code of this paper useful for your research, please consider citing us:

To be updated
<!--
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
