## CycleGAN implement
---

#### Title
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

![alt text](./img/paper1.png "Novelty of pix2pix")


# Train cycleGAN with PyTorch

## Requirment
- Python                 3.7+
- torch                  1.7.1+cu110
- torchvision            0.8.2+cu110
- matplotlib             3.3.3
- numpy                  1.19.5
- Pillow                 8.1.0
- scikit-image           0.17.2
- scipy                  1.5.4
- tensorboard            2.4.1
- tensorboardX           2.1
- tqdm                   4.56.0

## Training

    $ python main.py --mode train 
                     --data_path ./monet2photo \
                     --ckpt_path ./ckpt \
                     --result_path ./result \
                     --gpu 0
---

* Set 
* Hyperparameters were written to **arg.txt** under the **[log directory]**.



## Test
    $ python main.py --mode test 
                     --data_path ./monet2photo \
                     --ckpt_path ./ckpt \
                     --result_path ./result \
                     --gpu 0
---

* To test using trained network, set **ckpt_path** defined in the **train** phase.
* Generated image will be saved as ** sample.jpg ** in the ** [result directory] ** folder.


## Results
