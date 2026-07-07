


# STAM
This is the program page of our latest STAM approach for Face Video Super-Resolution(FVSR).  


https://github.com/user-attachments/assets/1d052dcd-ba65-4f4f-bddf-2aee0f9228eb

### Environment

• Python 3.8.19

• PyTorch >= 2.4.1

• RTX 3090

### Datasets
This study uses publicly available datasets. The CelebV-HQ used for training and testing is available at https://drive.google.com/drive/folders/19DLr27P9xMOTn_W6hxpxxm8_5jJoX-nR.

VFHQ also used for tesing is available at https://liangbinxie.github.io/projects/vfhq/

### Pretrained model
you can get pretrained model from https://drive.google.com/file/d/1_m00UD33QhmoDJhLTYNiJ5zKobyzWojw/view?usp=drive_link

### Training
Train the model by following the command lines below.

--python train.py --data_path 

Details:

--data_path: [should be filled in the directory to the training dataset]
--ckpt_path: [the location of your training model]

### Inference
After the training you can run the following command to FVSR for evaluation. 

python eval.py --data_path 
