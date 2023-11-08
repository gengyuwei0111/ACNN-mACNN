# ACNN-mACNN
# A Deep Learning Method for the Dynamics of Classic and Conservative Allen-Cahn Equations Based on Fully-Discrete Operators
Code repository for the paper:  
**A Deep Learning Method for the Dynamics of Classic and Conservative Allen-Cahn Equations Based on Fully-Discrete Operators**  
[Yuwei Geng](https://sc.edu/study/colleges_schools/artsandsciences/mathematics/our_people/directory/geng-yuwei.php), [Yuankai Teng](https://slooowtyk.github.io), [Zhu Wang](https://people.math.sc.edu/wangzhu) and [Lili Ju](https://people.math.sc.edu/ju)
<br>
Journal of Computational Physics [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999123006848)]
<br>



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

```bibtex
@article{geng2023deep,
  title={A Deep Learning Method for the Dynamics of Classic and Conservative Allen-Cahn Equations Based on Fully-Discrete Operators},
  author={Geng, Yuwei and Teng, Yuankai and Wang, Zhu and Ju, Lili},
  journal={Journal of Computational Physics},
  pages={112589},
  year={2023},
  publisher={Elsevier}
}
```

