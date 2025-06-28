# 📂 Data Preparation

## 1️⃣ Download the Dataset
The framework was evaluated using the **publicly available GastroHUN UGI video endoscopy dataset**.

🔗 **Dataset:** Available on [Figshare](https://doi.org/10.6084/m9.figshare.27308133)

<!--<img src="https://s3-eu-west-1.amazonaws.com/pfigshare-u-previews/223967/preview.jpg" width="600">-->

## 2️⃣ Extract Dataset Files
📌 Unzip Video Endoscopies

Extract the video endoscopy files into the `data/` folder.

📁 VideoEndoscopies Archive List:
- Videoendoscopies_Group1_Patients_7-103.zip
- Videoendoscopies_Group2_Patients_104-133.zip
- Videoendoscopies_Group3_Patients_136-202.zip
- Videoendoscopies_Group4_Patients_203-248.zip
- Videoendoscopies_Group5_Patients_250-301.zip
- Videoendoscopies_Group6_Patients_302-354.zip
- Videoendoscopies_Group7_Patients_355-387.zip

📌 Unzip Stomach Site Labeled Sequences

Extract the labeled sequences into the `data/` folder.

📁 Labeled Sequences Archive List:

- Labeled_Sequences_Group1_Patients_7-113.zip
- Labeled_Sequences_Group2_Patients_115-191.zip
- Labeled_Sequences_Group3_Patients_192-229.zip
- Labeled_Sequences_Group4_Patients_231-273.zip
- Labeled_Sequences_Group5_Patients_274-318.zip
- Labeled_Sequences_Group6_Patients_319-375.zip
- Labeled_Sequences_Group7_Patients_376-387.zip

Note 🗈: Unzip the Stomach Site Labeled Sequences in data folder.

# 🛠 Preprocessing 
🎥 Convert Videos from 30 FPS to 15 FPS

To normalize the frame rate of video endoscopies, use ffmpeg.

🚀 Run the Video Normalization Script

1️⃣ Open a terminal and navigate to the directory containing `convert_video_to_15fps.py`.

2️⃣ Define the required paths and run the script:

    ```bash
    # Set your paths
    OUTPUT_DIR='../data/15FPS'
    DATA_PATH='../data/Videoendoscopies'
    DATAFRAME='../official_splits/videoendoscopy-metadata.json'

    # Run the conversion script
    python convert_video_to_15fps.py \
      --path_data ${DATA_PATH} \
      --output_dir ${OUTPUT_DIR} \
      --dataframe ${DATAFRAME} \
    ```

⚠️ Important Notes

✅ Ensure the following paths exist before running the script:

 - `DATA_PATH` → Directory containing raw video endoscopies.
 - `DATAFRAME` → JSON file with metadata for video processing.