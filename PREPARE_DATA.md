# Prepare Dataset for ENM-MCL

## Our recorded in-house dataset by a KUKA youBot platform
Downloading in-house dataset from our server:
```shell
cd enm-mcl && mkdir data && cd data
wget https://www.ipb.uni-bonn.de/html/projects/kuang2025icra/ipblab.zip
unzip ipblab.zip
```

For each sequence, we provide :
- seq_{id}.bag: the ROS bag format, include raw odometer reading and raw lidar reading.
- seq_{id}.json: include raw odometer reading, ground-truth poses, and raw lidar reading.
- seq_{id}_gt_pose: the ground-truth poses in TUM format (for evaluation with evo). 

Besides, there are also some configuration files are provided:
- lidar_info.json: the parameters of the 2D LiDAR sensor.
- occmap.npy: the pre-built occupancy grid map.
- occmap.png: the visualization of the occupancy grid map.
- b2l.txt: the transformation from the lidar link to robot's base link 

The final data structure should look like
```
data/
├── ipblab/
│   ├── loc_test/
│   │   ├──b2l.txt
│   │   ├── test1/
│   │   │   ├──seq_1.bag
│   │   │   ├──seq_1.json
│   │   │   ├──seq_1_gt_pose.txt
│   │   ├── ...
│   ├── mapping/
│   │   ├── train.json
│   │   ├── test.json
│   │   ├── val.json
│   ├──lidar_info.json
│   ├──occmap.npy
│   ├──occmap.png
```
