

<video 
  src="https://github.com/user-attachments/assets/f6b0482d-9ba9-4857-ac14-3c9a0cdb3ce0" 
  width="700" 
  autoplay 
  loop 
  muted 
  playsinline>
</video>

# STAM
This is the program page of our latest STAM approach for Face Video Super-Resolution(FVSR).   Version 1


### Dataset
you can download training data from https://drive.google.com/drive/folders/19DLr27P9xMOTn_W6hxpxxm8_5jJoX-nR

### Training
Train the model by following the command lines below. (Waiting update)

--python train.py

Details:

--data_path: [should be filled in the directory to the training dataset]
--ckpt_path: [the location of your training model]

### Inference
After the training you can run the following command to FVSR for evaluation. (Waiting update)

python eval.py
