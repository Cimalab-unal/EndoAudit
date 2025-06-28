# Training: Single-Fame Classification
## 1. **Run training single-frame classification:** 
Open a terminal and navigate to the directory containing `mpl_embedding.py` and define the parameters:

    ```bash
    FEATURE_PATH= '..\features\15 FPS\Pretrained_Weights_GastroHUN'
    OUTPUT_DIR='output\9.0'
    DATA_TRAIN='..\experimental_setup\Single-Frame\train.json'
    DATA_VALIDATION='..\experimental_setup\Single-Frame\validation.json'

    !python mpl_embedding.py \
    --nb_classes 4 \
    --num_epochs 50 \
    --early_stopping 10 \
    --lr 0.001 \
    --batch_size 128 \
    --features_path ${FEATURE_PATH}  \    
    --dataframe_train ${DATA_TRAIN}  \
    --dataframe_validation ${DATA_VALIDATION}  \    
    --output_dir ${OUTPUT_DIR} 
    ```

### Note:
1. Check the existence of the paths of: 
   - FEATURE_PATH: Select the folder containing the features (e.g CNN ImageNet, CNN endoscopy).
   - DATA_TRAIN
   - DATA_VALIDATION 

## 2. **Run testing single-frame classification:** 
Open the Jupyter Notebook `test_single_frame_classification.ipynb` and define the parameters:

# Training: Multi-Fame Classification

## 1. **Run training multiframe-frame classification:**  One Attention Layer
Open a terminal and navigate to the directory containing `attention_training.py` and define the parameters:

    ```bash
    FEATURE_PATH= '..\features\15 FPS\Pretrained_Weights_GastroHUN'
    OUTPUT_DIR='output\9.0'
    DATA_TRAIN='..\experimental_setup\Single-Frame\train.json'
    DATA_VALIDATION='..\experimental_setup\Single-Frame\validation.json'

    !python attention_training.py \
    --nb_classes 4 \
    --num_epochs 50 \
    --early_stopping 10 \
    --lr 0.001 \
    --batch_size 128 \
    --features_path ${FEATURE_PATH}  \    
    --dataframe_train ${DATA_TRAIN}  \
    --dataframe_validation ${DATA_VALIDATION}  \    
    --output_dir ${OUTPUT_DIR} 
    ```
## 2. **Run training multiframe-frame classification:**  ViT-Base
Open a terminal and navigate to the directory containing `vit-base_training.py` and define the parameters:

    ```bash
    FEATURE_PATH= '..\features\15 FPS\Pretrained_Weights_GastroHUN'
    OUTPUT_DIR='output\9.0'
    DATA_TRAIN='..\experimental_setup\Single-Frame\train.json'
    DATA_VALIDATION='..\experimental_setup\Single-Frame\validation.json'

    !python attention_training.py \
    --nb_classes 4 \
    --num_epochs 50 \
    --early_stopping 10 \
    --lr 0.001 \
    --batch_size 128 \
    --features_path ${FEATURE_PATH}  \    
    --dataframe_train ${DATA_TRAIN}  \
    --dataframe_validation ${DATA_VALIDATION}  \    
    --output_dir ${OUTPUT_DIR} 
    ```

### Note:
1. Check the existence of the paths of: 
   - FEATURE_PATH: Select the folder containing the features (e.g CNN ImageNet, CNN endoscopy).
   - DATA_TRAIN
   - DATA_VALIDATION 