# Segment-Anything Fine-Tuning for Specific Type Images

This is the repository for assignment2 of course "Image Processing and Analysis" of Westlake University, 
which is an implementation of a project that annotates images using Segment-Anything (SAM) and further fine-tunes SAM on the specific type images.

## Quickstart

### 1. Setup environment

```
git clone https://github.com/lizhiqi49/SAM-Finetune.git
cd SAM-Finetune
pip install -r requirements.txt
```

### 2. Get access to pretrained SAM model

This repository use pretrained SAM model supported in library `transformers`, you can get access to the pretrained model on `huggingface-hub`.

There are three versions of released SAM model: `sam-vit-base`, `sam-vit-large` and `sam-vit-huge`. They are named according to their number of parameters.

You can specify one of them as the base SAM model for mask generation and fine-tuning in the scripts, such as, `facebook/sam-vit-base`. Or you can download the model to your host.

For more details please refer to [huggingface's official documentation](https://huggingface.co/docs/transformers/main/en/model_doc/sam)



## Usage of mask generation tool

In the project directory, you can use the mask generation tool `mask_generation.py` to annotate your image datas powered by pretrained SAM model and custom your prompts (bounding boxes, keypoints) with your mouse and keyboard.

When running the script, this is how it works:

Step 1. The image will be displayed in an opencv window, you should custom the bounding box which indicates the region of interest where you want to perform segmentation by sliding your mouse. You can repeatly select the bounding box until you double-press `ENTER` on your keyboard;

Step2. Then you need to pick those keypoints within the bbox to give the model more exact conditions. Use left button of your mouse to tell the model where you want the segmentation result to contain and right button for opposite. When finished, press `q` on your keyboard to exit.

Step3. The model will generate masks based on all your provided prompts. When it finished, three masked images will be displayed and you should choose the one you are most satisfied with by inputing its index (1/2/3) in your command line.

Step4. Finally, the mask and ROI will be saved on your configured output directory.

