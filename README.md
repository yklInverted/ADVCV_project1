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
-
