testacs2:
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Set5/HR --test_lr_folder Test_Datasets/Set5/LR/ --output_folder results/ACS2Set5/ --checkpoint checkpoints/acs2_e1000.pth --acs 2
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Set14/HR --test_lr_folder Test_Datasets/Set14/LR/ --output_folder results/ACS2Set14/ --checkpoint checkpoints/acs2_e1000.pth --acs 2
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/BSD100/HR/ --test_lr_folder Test_Datasets/BSD100/LR/ --output_folder results/ACS2BSD100/ --checkpoint checkpoints/acs2_e1000.pth --acs 2
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Urban100/HR/ --test_lr_folder Test_Datasets/Urban100/LR/ --output_folder results/ACS2Urban100/ --checkpoint checkpoints/acs2_e1000.pth --acs 2

testacs3:
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Set5/HR --test_lr_folder Test_Datasets/Set5/LR/ --output_folder results/ACS3Set5/ --checkpoint checkpoints/acs3_e1000.pth --acs 3
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Set14/HR/ --test_lr_folder Test_Datasets/Set14/LR/ --output_folder results/ACS3Set14/ --checkpoint checkpoints/acs3_e1000.pth --acs 3
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Urban100/HR/ --test_lr_folder Test_Datasets/Urban100/LR/ --output_folder results/ACS3Urban100/ --checkpoint checkpoints/acs3_e1000.pth --acs 3
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/BSD100/HR/ --test_lr_folder Test_Datasets/BSD100/LR/ --output_folder results/ACS3BSD100/ --checkpoint checkpoints/acs3_e1000.pth --acs 3

testacs4:
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Set5/HR --test_lr_folder Test_Datasets/Set5/LR/ --output_folder results/ACS4Set5/ --checkpoint checkpoints/acs4_e1000.pth --acs 4
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Set14/HR/ --test_lr_folder Test_Datasets/Set14/LR/ --output_folder results/ACS4Set14/ --checkpoint checkpoints/acs4_e1000.pth --acs 4
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/BSD100/HR/ --test_lr_folder Test_Datasets/BSD100/LR/ --output_folder results/ACS4BSD100/ --checkpoint checkpoints/acs4_e1000.pth --acs 4
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Urban100/HR/ --test_lr_folder Test_Datasets/Urban100/LR/ --output_folder results/ACS4Urban100/ --checkpoint checkpoints/acs4_e1000.pth --acs 4

timetest:
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Set14/HR --test_lr_folder Test_Datasets/Set14/LR/ --output_folder results/ACS2Set14/ --checkpoint checkpoints/acs2_e1000.pth --acs 2
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Set5/HR --test_lr_folder Test_Datasets/Set5/LR/ --output_folder results/ACS2Set5/ --checkpoint checkpoints/acs2_e1000.pth --acs 2
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/BSD100/HR/ --test_lr_folder Test_Datasets/BSD100/LR/ --output_folder results/ACS2BSD100/ --checkpoint checkpoints/acs2_e1000.pth --acs 2
	python3 test_IMDN_AS.py --test_hr_folder Test_Datasets/Urban100/HR/ --test_lr_folder Test_Datasets/Urban100/LR/ --output_folder results/ACS2Urban100/ --checkpoint checkpoints/acs2_e1000.pth --acs 2

trainacs2:
	python3 train_IMDN_AS.py --acs 2 --root .. -nEpochs 1000
trainacs3:
	python3 train_IMDN_AS.py --acs 3 --root .. -nEpochs 1000
trainacs4:
	python3 train_IMDN_AS.py --acs 4 --root .. -nEpochs 1000

testall: testacs2 testacs3 testacs4
