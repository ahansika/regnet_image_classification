# regnet_image_classification
@@ -27,38 +27,40 @@
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz
```
### 2. Train Model
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
```
CIFAR-10 and CIFAR-100 are automatically download and train. In order to use a different dataset you need to customize [data_utils.py](./utils/data_utils.py).
The default batch size is 512. When GPU memory is insufficient, you can proceed with training by adjusting the value of `--gradient_accumulation_steps`.
Also can use [Automatic Mixed Precision(Amp)](https://nvidia.github.io/apex/amp.html) to reduce memory usage and train faster
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
```
## Results
To verify that the converted model weight is correct, we simply compare it with the author's experimental results. We trained using mixed precision, and `--fp16_opt_level` was set to O2.

### imagenet-21k
* [**tensorboard**](https://tensorboard.dev/experiment/XvHOVNtMS02KOrmxOOJAEg/#scalars)
* [**tensorboard**](https://tensorboard.dev/experiment/Oz9GmmQIQCOEr4xbdr8O3Q)

|    model     |  dataset  | resolution | acc(official) | acc(this repo) |  time   |
|:------------:|:---------:|:----------:|:-------------:|:--------------:|:-------:|
|   ViT-B_16   | CIFAR-10  |  224x224   |       -       |     0.9908     | 3h 13m  |
|   ViT-B_16   | CIFAR-10  |  384x384   |    0.9903     |     0.9906     | 12h 25m |
|   ViT_B_16   | CIFAR-100 |  224x224   |       -       |     0.923      |  3h 9m  |
|   ViT_B_16   | CIFAR-100 |  384x384   |    0.9264     |     0.9228     | 12h 31m |
| R50-ViT-B_16 | CIFAR-10  |  384x384   |     0.99      |      WIP       |         |
| R50-ViT-B_16 | CIFAR-100 |  384x384   |    0.9231     |      WIP       |         |
| R50-ViT-B_16 | CIFAR-10  |  224x224   |       -       |     0.9892     | 4h 23m  |
| R50-ViT-B_16 | CIFAR-10  |  384x384   |     0.99      |     0.9904     | 15h 40m |
| R50-ViT-B_16 | CIFAR-100 |  224x224   |       -       |     0.9231     | 4h 18m  |
| R50-ViT-B_16 | CIFAR-100 |  384x384   |    0.9231     |     0.9197     | 15h 53m |
|   ViT_L_32   | CIFAR-10  |  224x224   |       -       |     0.9903     | 2h 11m  |
|   ViT_L_32   | CIFAR-100 |  224x224   |       -       |     0.9273     |  2h 9m  |
|   ViT_L_32   | CIFAR-100 |  224x224   |       -       |     0.9276     |  2h 9m  |
|   ViT_H_14   | CIFAR-100 |  224x224   |       -       |      WIP       |         |


  4 changes: 2 additions & 2 deletions4  
train.py
@@ -269,11 +269,11 @@ def main():
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=500, type=int,
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=100, type=int,
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
