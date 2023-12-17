### Note: The Test Datasets have been included in this assignment to make testing quicker which is why it is so large.

# Downloading and configuring the training data set

We got the DIV2K dataset from https://data.vision.ee.ethz.ch/cvl/DIV2K/. More specifically, we used the data from the DIV2K_train_HR and DIV2K_train_LR_bicubic/X2 folders from the site below for our training data set.

To download the datasets mentioned above add the name of one of the four subfolders to the end of the link below.
https://data.vision.ee.ethz.ch/cvl/ + [subfolder]

eg. https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

Original Structure
```
└── IMDN
    ├── FLOPs
    │   └── __pycache__
    ├── TF_model
    ├── Test_Datasets
    │   ├── RealSR
    │   │   ├── ValidationGT
    │   │   └── ValidationLR
    │   ├── Set5
    │   └── Set5_LR
    │       ├── x2
    │       ├── x3
    │       └── x4
    ├── checkpoints
    ├── data
    ├── images
    ├── model
    │   └── __pycache__
    └── scripts
```

Once you have downloaded all of the folders for the training set above, move them so they are outside of the IMDN directory so the structure looks like what you see below. 

```
.
├── DIV2K_train_LR_bicubic/X2.zip
├── DIV2K_train_HR.zip
├── IMDN
```

### Using whatever you like unzip the folders. 

## Formatting the images

Before you can use the images, you have to convert them to png with the following command.

```bash
python scripts/png2npy.py --pathFrom /path/to/DIV2K/ --pathTo /path/to/DIV2K_decoded/
```

Save the dataset so it has the following structure and names as the tree below.

```
.
├── DIV2K_HR_decoded
├── DIV2K_LR_x2_decoded
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   ├── X2
├── DIV2K_valid_HR
├── IMDN
```

## Setting up the IMDN env
Using the env.yml file to create the conda enviornment in order to run the IMDN code

```bash
conda env create --file env.yml
```

Activate the conda environment
```bash
conda activate imdn
```

## Training the model

Run the following command from the Makefile below to train the ACS 2, ACS 3 and ACS 4 models.

```bash
make trainacs2
```
```bash
make trainacs3
```
```bash
make trainacs4
```

## The Test Datasets have been included in this to reduce the amount of time it takes to get the model to work.

They can be found in the Test_Datasets folder and contain LR (Low Resolution) and HR (High Resolution) images. The tree structure of the models is below.

```
.
├── BSD100
│   ├── HR
│   └── LR
├── RealSR
│   ├── ValidationGT
│   └── ValidationLR
├── RealSR_decoded
│   ├── TrainGT
│   └── TrainLR
├── Set14
│   ├── HR
│   └── LR
├── Set5
│   ├── HR
│   └── LR
└── Urban100
    ├── HR
    └── LR
```

## Testing the model

Run the commands from the Makefile to test the models we trained which can be found in the checkpoints directory. If you want to test your own models you will need to update the checkpoint argument in the Makefile with the name of your pth file.

If you are having trouble getting the testing to work, you can train your own model and then test it yourself by changing the README.md which works. If the ones from checkpoints don't work, try grabbing them from the roughwork. If that doens't work train your own and test that.

```bash
make testacs2
```
```bash
make testacs3
```
```bash
make testacs4
```

OR this to test them all at once

```bash
make testall
```

## Extra Info: Testing the provided IMDN model and testing a IMDN_AS model

Feel free to add it to a Makefile if you want.

```bash
# Set5 x2 IMDN
python3 test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/Set5/x2 --checkpoint checkpoint/NAMEOFMODEL.pth --upscale_factor 2
# RealSR IMDN_AS
python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/RealSR/ValidationGT --test_lr_folder Test_Datasets/RealSR/ValidationLR/ --output_folder results/RealSR --checkpoint checkpoint/NAMEOFMODEL.pth
```
