<p align="center">
  <h1 align="center"> ENM-MCL: Improving Indoor Localization Accuracy by Using an Efficient Implicit Neural Map Representation </h1>

  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/haofei-kuang/"><strong>Haofei Kuang</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/yue-pan/"><strong>Yue Pan</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/xingguang-zhong/"><strong>Xingguang Zhong</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/louis-wiesmann/"><strong>Louis Wiesmann</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/jens-behley/"><strong>Jens Behley</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a>
  </p>
  <p align="center"><a href="https://www.ipb.uni-bonn.de"><strong>University of Bonn</strong></a>
</p>

![enm_mcl_teaser](https://github.com/user-attachments/assets/50ae6a06-a55b-4013-9c3c-100e05da92a1)

*Accuarate and efficient indoor global localization with ENM-MCL in the Lab of Uni.Bonn.*

---

## Abstract
<details>
  <summary>[Details (click to expand)]</summary>
Globally localizing a mobile robot in a known map is often a foundation for enabling robots to navigate and operate autonomously. 
In indoor environments, traditional Monte Carlo localization based on occupancy grid maps is considered the gold standard, 
but its accuracy is limited by the representation capabilities of the occupancy grid map.
In this paper, we address the problem of building an effective map representation that allows to accurately perform probabilistic global localization.
To this end, we propose an implicit neural map representation that is able to capture positional 
and directional geometric features from 2D LiDAR scans to efficiently represent the environment and 
learn a neural network that is able to predict both, the non-projective signed distance and 
a direction-aware projective distance for an arbitrary point in the mapped environment.
This combination of neural map representation with a light-weight neural network allows us to design 
an efficient observation model within a conventional Monte Carlo localization framework for pose estimation of a robot in real time.
We evaluated our approach to indoor localization on a publicly available dataset for global localization and 
the experimental results indicate that our approach is able to more accurately localize a mobile robot than 
other localization approaches employing occupancy or existing neural map representations.
In contrast to other approaches employing an implicit neural map representation for 2D LiDAR localization, 
our approach allows to perform real-time pose tracking after convergence and near real-time global localization.
</details>


## Installation

The code was tested with Ubuntu 22.04 with:
- python version **3.10**.
- pytorch version **2.6.0** with **CUDA 11.8**

We recommend using **Conda** to install the dependencies:
```shell
cd ~ && git clone https://github.com/PRBonn/enm-mcl.git
cd ~/enm-mcl
conda env create -f environment.yml
conda activate enmmcl
```
Or install manually:
```shell
conda create --name enmmcl python=3.10
conda activate enmmcl

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib scipy open3d opencv-python
pip install evo --upgrade --no-binary evo
conda install -c conda-forge pybind11
```

Follow the instructions to compile the motion model and resampling modules:
```shell
cd ~/enm-mcl/localization/mcl/ && conda activate enmmcl
make -j4
```

## Dataset

Please refer to [PREPARE_DATA](PREPARE_DATA.md) to prepare the datasets

## EfficientNeuralMap Mapping

- Run ENM training on the mapping sequence:
  ```shell
  cd ~/enm-mcl && conda activate enmmcl
  python run_mapping.py --config_file configs/mapping_enm.yaml
  ```

- Evaluate the trained ENM model:
  ```shell
  cd ~/enm-mcl && conda activate enmmcl
  python eval_map.py --config_file configs/mapping_enm.yaml
  ```

## Global Localization with ENM-MCL
- Run ENM-MCL on a test sequence:
  ```shell
  cd ~/enm-mcl && conda activate enmmcl
  python run_localization.py --config_file configs/global_localization/loc_config_test1.yaml
  ```
  Results will be saved to the `results/` folder.

### To reproduce the results of the paper:
- Download the pre-trained ENM model from [here](https://www.ipb.uni-bonn.de/html/projects/kuang2025icra/enm_map.pth) and save it to the `results/` folder.
  ```shell
  cd ~/enm-mcl && mkdir results && cd results
  wget https://www.ipb.uni-bonn.de/html/projects/kuang2025icra/enm_map.pth
  ```
- Run the following command to evaluate the ENM-MCL on the test sequences:
  ```shell
  cd ~/enm-mcl && conda activate enmmcl
  ./run_loc.sh
  ```

## Citation
If you use this library for any academic work, please cite our original [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/kuang2025icra.pdf).

```bibtex
@inproceedings{kuang2025icra,
  author = {H. Kuang and Y. Pan and X. Zhong and L. Wiesmann and J. Behley and Stachniss, C.},
  title = {{Improving Indoor Localization Accuracy by Using an Efficient Implicit Neural Map Representation}},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2025}
}
```

## Contact
If you have any questions, please contact:

- Haofei Kuang {[haofei.kuang@igg.uni-bonn.de]()}


## Related Projects

[IR-MCL (RAL 23)](https://github.com/PRBonn/ir-mcl): Implicit Representation for Monte Carlo Localization

[LocNDF (RAL 23)](https://github.com/PRBonn/LocNDF): Neural Distance Field Mapping for Robot Localization

[SHINE-Mapping (ICRA 23)](https://github.com/PRBonn/SHINE_mapping): Large-Scale 3D Mapping Using Sparse Hierarchical Implicit Neural Representations


## Acknowledgment

This work has partially been funded by:

- the **Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)** under Germany's Excellence Strategy, EXC-2070 -- 390732324 -- PhenoRob,
- and by the **German Federal Ministry of Education and Research (BMBF)** in the project "Robotics Institute Germany" under grant No. 16ME0999.
