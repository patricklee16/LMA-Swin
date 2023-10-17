# LMA-Swin
## Introduction
In this repository, we offer the code of LMA-Swin, a semantic segmentation model based on PyTorch which combines Convolutional Neural Network (CNN) features with Transformer features through modulation, addressing the limitations of Transformer-based models in preserving fine-grained details and local edge segmentation.
You can download the trained model weights on [BaiduDisk](https://pan.baidu.com/s/1dvIplebkSeyMA9jU2g7ffw).   Code: lmas

## Network
![image](https://github.com/patricklee16/LMA-Swin/assets/51188249/a16c6671-f9c3-47fb-92e1-49159111dda5)

## Dependency Installation
```
conda create -n airs python=3.8
conda activate airs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r GeoSeg/requirements.txt
```

## Data Preparation

Please download the datasets from the official website and prepare the data following the following file structure:
```
airs
├── GeoSeg (code)
├── pretrain_weights (pretrained weights of backbones)
├── model_weights (save the model weights trained on datasets)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── train_val (processed)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
```

## Training
```
python GeoSeg/train_supervision.py -c GeoSeg/config/vaihingen/lmaswin.py
```
"-c" means the path of the config, use different config to train different models.

## Testing
**Vaihingen**
```
python GeoSeg/vaihingen_test.py -c GeoSeg/config/vaihingen/dcswin.py -o fig_results/vaihingen/dcswin --rgb -t 'd4'
```
**Potsdam**
```
python GeoSeg/potsdam_test.py -c GeoSeg/config/potsdam/dcswin.py -o fig_results/potsdam/dcswin --rgb -t 'lr'
```

## Visualization Results
![image](https://github.com/patricklee16/LMA-Swin/assets/51188249/566d17d1-2e49-4542-89df-99a3a620718e)

![image](https://github.com/patricklee16/LMA-Swin/assets/51188249/9cb65b49-8ef5-411a-b340-dc7bcb4f956e)

![image](https://github.com/patricklee16/LMA-Swin/assets/51188249/7c05728e-77ce-4d53-9af2-b891c8bafa48)

![image](https://github.com/patricklee16/LMA-Swin/assets/51188249/68222a4a-6d41-4dd9-bb25-0c6b99b8a201)

## Acknowledgement
Many thanks for the [GeoSeg](https://github.com/WangLibo1995/GeoSeg)'s contributions to LMA-Swin.
