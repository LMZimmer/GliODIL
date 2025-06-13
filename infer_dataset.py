import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from parsing import LongitudinalDataset


def convert_tumorseg_labels(seg_dir, outfile):

    seg = nib.load(seg_dir)
    aff = np.eye(4)
    seg_data = np.rint(seg.get_fdata()).astype(np.int32)

    # GLIODIL:      1: non_enhancing, 3: edema, 4: enhancing
    # BRATS (new):  1: non_enhancing, 2: edema, 3: enhancing
    seg_data[(seg_data == 2) | (seg_data == 3)] += 1
    seg_new = nib.Nifti1Image(seg_data, affine=aff)
    nib.save(seg_new, outfile)


if __name__ == "__main__":
    # Example:
    # python infer_dataset.py -cuda_device 0
    # nohup python -u infer_dataset.py -dataset gliodil -cuda_device 0 > tmp_gliodil.out 2>&1 &
    # nohup python -u infer_dataset.py -dataset lumiere -cuda_device 1 > tmp_lumiere.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument("-dataset", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    dataset = None
    if args.dataset == "rhuh":
        RHUH_GBM_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/rhuh.json")
        rhuh_root = "/home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM"
        dataset = LongitudinalDataset(dataset_id="RHUH", root_dir=rhuh_root)
        dataset.load(RHUH_GBM_DIR)
    elif args.dataset == "upenn":
        UPENN_GBM_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/upenngbm.json")
        upenn_gbm_root = "/home/home/lucas/data/UPENN-GBM/UPENN-GBM"
        dataset = LongitudinalDataset(dataset_id="UPENN_GBM", root_dir=upenn_gbm_root)
        dataset.load(UPENN_GBM_DIR)
    elif args.dataset == "gliodil":
        GLIODIL_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/gliodil.json")
        gliodil_root = "/mnt/Drive2/lucas/datasets/GLIODIL"
        dataset = LongitudinalDataset(dataset_id="GLIODIL", root_dir=gliodil_root)
        dataset.load(GLIODIL_DIR)
    elif args.dataset == "lumiere":
        LUMIERE_DIR = Path("/home/home/lucas/projects/gbm_bench/gbm_bench/data/datasets/lumiere.json")
        lumiere_root = "/mnt/Drive2/lucas/datasets/LUMIERE/Imaging"
        dataset = LongitudinalDataset(dataset_id="LUMIERE", root_dir=lumiere_root)
        dataset.load(LUMIERE_DIR)
    elif args.dataset == "ivygap":
        pass
    if dataset is None:
        raise ValueError(f"Dataset {args.dataset} not implemented.")

    starting_ind = 0
    print(starting_ind)
    for patient_ind, patient in enumerate(dataset.patients[starting_ind:]):  # hung at 5 for lumiere
        print(f"Predicting {patient_ind}/{len(dataset.patients)}...")

        for exam in patient["exams"]:
            if exam["timepoint"] != "preop":
                continue
            
            if args.dataset == "gliodil":
                patient_dir = exam["t1c"].parent / "preop"
            else:
                patient_dir = exam["t1c"].parent
            print(patient_dir)

            wmPath = str(patient_dir / "processed/tissue_segmentation/wm_pbmap.nii.gz")
            gmPath = str(patient_dir / "processed/tissue_segmentation/gm_pbmap.nii.gz")
            tumorSegmentationPath = str(patient_dir / "processed/tumor_segmentation/tumor_seg.nii.gz")
            savePath = str(patient_dir / "processed/growth_models/gliodil")
            os.makedirs(savePath, exist_ok=True)
            logfile = str(patient_dir / "processed/growth_models/gliodil/logfile.log")

            tumorSegmentationPath_134 = str(patient_dir / "processed/growth_models/gliodil/tumor_seg_134.nii.gz")
            convert_tumorseg_labels(seg_dir=tumorSegmentationPath, outfile=tumorSegmentationPath_134)


            cmd = f'USEGPU=1 CUDA_VISIBLE_DEVICES={args.cuda_device} /home/home/lucas/projects/dockerize/GliODIL/GliODIL.py --outdirectory "{savePath}" --optimizer adamn --lambda_pde_multiplier 1.0 --Nt 192 --Nx 48 --Ny 48 --Nz 48 --days 100 --history_every 1000 --report_every 1000 --epochs 9000 --plot_every 3000 --save_solution y --final_print y --multigrid 1 --save_forward odil_res --save_forward2 full_trim_Gauss --initial_guess forward_character_dice_breaking --seg_path "{tumorSegmentationPath_134}" --wm_path "{wmPath}"  --gm_path "{gmPath}" --pet_path ""'
            print(cmd)
    
            try:
                os.system(cmd)
                pred_file = str(patient_dir / "processed/growth_models/gliodil/192_48_48_48_solution.nii")
                new_file = str(patient_dir / "processed/growth_models/gliodil/gliodil_pred.nii.gz")
                pred = nib.load(pred_file)
                nib.save(pred, new_file)
            except Exception as e:
                print(f"Exception for {patient_ind}: e")
    print("Done.")
