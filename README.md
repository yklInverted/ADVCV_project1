# ADVCV_project1

### K Li


## Prepare

- Download Dataset

  Download and extract data, you should have your dataset folder at:
    ADVCV_Project1/__MACOSX ADVCV_Project1/FashionDataset 

- Create Conda Environment
    ```
    cd code
    conda env create -f env.yaml
    conda activate advcv
    ```
- Run with checkpoint:
  
    ```
    cd code
    python train.py --load_checkpoint
    ```
    if you want to retrain the model without using existing checkpoint:
    ```
    cd code
    python train.py
    ```

## 注意事项
- 由于我在deadline前最后一周得了新冠，请了五天病假直到29号，而codalab于24号关闭无法继续提交，所以我无法计算我最后的test score，只能把最后输出的txt文件与其他需要提交的文件一起提交。麻烦您帮我核算最后的test结果。最后输出的txt文件保存在当前目录下，命名为'result.txt'
