## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading models automatically from
the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
  
python ./yolov5/detect.py --source ./datasets/images/IMG_20210701_141111.jpg --weights ./trained_model/yolov5s6_v1.pt --img 1280

python ./yolov5/detect.py --source ./datasets/images/IMG_20210701_141111.jpg --weights ./trained_model/yolov5s_v1.pt --img 640
```

</details>

<details open>
<summary>Training</summary>

Run commands below to reproduce results
on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on
first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the
largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
  
```
dossier_projet/
  -trained_model/
  -mobile_model/
  -datasets/
        images/
        labels/
  -yolov5/
```  

```bash
$ python ./yolov5/train.py --img 640 --batch 64 --epochs 10 --data ./yolov5/data/poubelle.yaml --weights ./yolov5/yolov5s.pt
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>  


<details open>
<summary>Export</summary>

Les commandes sont pour exporter le modele vers une version mobile
  
```bash
$ python ./yolov5/export.py --weights ./trained_model/yolov5s6_v1_1280.pt --img-size 1280 
  --include torchscript --optimize --output ./mobile_model/yolov5s6_v1_1280.ptl
```


</details>  

<details>
<summary>Tutorials</summary>

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️
  RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Supervisely Ecosystem](https://github.com/ultralytics/yolov5/issues/2518)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [TorchScript, ONNX, CoreML Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)

</details>
## <div align="center">Why YOLOv5</div>

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure (click to expand)</summary>

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313219-f1d70e00-9af5-11eb-9973-52b1f98d321a.png"></p>
</details>
<details>
  <summary>Figure Notes (click to expand)</summary>

* GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size
  32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS.
* EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.
* **Reproduce** by
  `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>

### Pretrained Checkpoints

[assets]: https://github.com/ultralytics/yolov5/releases

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) |FLOPs<br><sup>640 (B)
|---                    |---  |---      |---      |---      |---     |---|---   |---
|[YOLOv5s][assets]      |640  |36.7     |36.7     |55.4     |**2.0** |   |7.3   |17.0
|[YOLOv5m][assets]      |640  |44.5     |44.5     |63.1     |2.7     |   |21.4  |51.3
|[YOLOv5l][assets]      |640  |48.2     |48.2     |66.9     |3.8     |   |47.0  |115.4
|[YOLOv5x][assets]      |640  |**50.4** |**50.4** |**68.8** |6.1     |   |87.7  |218.8
|                       |     |         |         |         |        |   |      |
|[YOLOv5s6][assets]     |1280 |43.3     |43.3     |61.9     |**4.3** |   |12.7  |17.4
|[YOLOv5m6][assets]     |1280 |50.5     |50.5     |68.7     |8.4     |   |35.9  |52.4
|[YOLOv5l6][assets]     |1280 |53.4     |53.4     |71.1     |12.3    |   |77.2  |117.7
|[YOLOv5x6][assets]     |1280 |**54.4** |**54.4** |**72.0** |22.4    |   |141.8 |222.9
|                       |     |         |         |         |        |   |      |
|[YOLOv5x6][assets] TTA |1280 |**55.0** |**55.0** |**72.0** |70.8    |   |-     |-

<details>
  <summary>Table Notes (click to expand)</summary>

* AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results
  denote val2017 accuracy.
* AP values are for single-model single-scale unless otherwise noted. **Reproduce mAP**
  by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
* Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a
  GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and
  includes FP16 inference, postprocessing and NMS. **Reproduce speed**
  by `python val.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45 --half`
* All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation).
* Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) includes reflection and scale
  augmentation. **Reproduce TTA** by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

## <div align="center">Contribute</div>

We love your input! We want to make contributing to YOLOv5 as easy and transparent as possible. Please see
our [Contributing Guide](CONTRIBUTING.md) to get started.

## <div align="center">Contact</div>

For issues running YOLOv5 please visit [GitHub Issues](https://github.com/ultralytics/yolov5/issues). For business or
professional support requests please visit [https://ultralytics.com/contact](https://ultralytics.com/contact).

<br>

<div align="center">
    <a href="https://github.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.linkedin.com/company/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://twitter.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://youtube.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.facebook.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.instagram.com/ultralytics/">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="3%"/>
    </a>
</div>
