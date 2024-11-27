# model's complexity between deepmd vs DPA

## 1. config
DPA 模型的路径
nas_96:/data/hhzheng/deepmd/nas_96/模型/DPA复现/ALMgCu_finetune/AlMgCu_fintune_3E6.pb
deepmd 模型的路径
/data/hhzheng/deepmd/github/AlMgCu-deepmd/01.train/graph_compressed.pb.
测试环境
qstation01
数据集：
基于mp-1200279材料，共有464 atoms, cubic的POSCAR文件，进行supercell。
共有：
'POSCAR_1**3'  'POSCAR_4**3'    'POSCAR_2**3'  'POSCAR_5**3'  'POSCAR_3**3'   
原子数分别为：
464 3712    12528   29696   58000

## 2. 测试
在qstation上使用2个GPU，可以测试到29696个原子数。

## 3. 比较
### 1. 模型参数配置
| 模型        | size    | descriptor    | sel   | fitting_net     |  
|------------|----------|---------------|-  ----|-----------------|
| deepmd     | 28M      | se_e2_a       |  120  | [240, 240, 240] |
| DPA        | 4.6M     | se_atten_v2   |  120  | [240, 240, 240] |


### 3. 推理的速度
### 4. 需要的资源，gpu， cpu , 内存
