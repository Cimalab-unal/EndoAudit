For all samples the data normalization was performed using the mean and standard deviation of the dataset.
  ```bash
  transform = transforms.Compose([
                  transforms.Resize(224,224), interpolation=Image.LANCZOS), 
                  transforms.ToTensor(),
                  transforms.Normalize([0.5990, 0.3664, 0.2769], [0.2847, 0.2190, 0.1772])
                  ]) 
  ```


# 1. Videoendoscopies
1. **ViT’s patch-based linear projection (16×16×3):** Run script open a terminal and navigate to the directory containing `linear_projection_features.py` and define the parameters:

    ```bash
    OUTPUT_DIR='..\data\15 FPS'
    DATA_PATH='..\data\Videoendoscopies'
    DATAFRAME='..\official_splits\videoendoscopy-metadata.json'

    !python linear_projection_features.py \
    --num_workers 0 \
    --batch_size 256 \
    --data_path ${DATA_PATH}  \
    --output_dir ${OUTPUT_DIR} \
    --dataframe  ${DATAFRAME}  
    ```
2. **ConvNeXt-Tiny pretrained on ImageNet:** Run script open a terminal and navigate to the directory containing `imagenet_features.py` and define the parameters:

    ```bash
    OUTPUT_DIR='..\data\15 FPS'
    DATA_PATH='..\data\Videoendoscopies'
    DATAFRAME='..\official_splits\videoendoscopy-metadata.json'

    !python imagenet_features.py \
    --num_workers 0 \
    --batch_size 256 \
    --data_path ${DATA_PATH}  \
    --output_dir ${OUTPUT_DIR} \
    --dataframe  ${DATAFRAME}  
    ```
3. **ConvNeXt-Tiny pretrained on Endoscopy:** Run script open a terminal and navigate to the directory containing `endoscopy_features.py` and define the parameters:
The **DATA_MODEL** is trained on the endoscopy dataset and can be downloaded from [Download](https://drive.google.com/uc?id=1ehpeF044ABRcwa6xMQ9zFRGfxWtuOav9)

    ```bash
    OUTPUT_DIR='..\data\15 FPS'
    DATA_PATH='..\data\Videoendoscopies'
    DATA_MODEL='..\best-model-val_f1_macro.ckpt'
    DATAFRAME='..\official_splits\videoendoscopy-metadata.json'

    !python endoscopy_features.py \
    --num_workers 0 \
    --batch_size 256 \
    --data_path ${DATA_PATH}  \
    --path_model ${DATA_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --dataframe  ${DATAFRAME}  
    ```


### Note:
1. Check the existence of the paths of: 
   - DATA_PATH
   - MODEL_PATH
   - DATAFRAME 

