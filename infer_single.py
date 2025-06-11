import os
import argparse
from pathlib import Path



def convert_tumorseg_labels(seg_dir):
    temp_dir = "/mnt/Drive2/lucas/tmp/tumorseg_134.nii.gz"

    seg = nib.load(seg_dir)
    aff = np.eye(4)
    seg_data = np.rint(seg.get_fdata()).astype(np.int32)

    # GLIODIL:      1: non_enhancing, 3: edema, 4: enhancing
    # BRATS (new):  1: non_enhancing, 2: edema, 3: enhancing
    seg_data[(seg_data == 2) | (seg_data == 3)] += 1
    seg_new = nib.Nifti1Image(seg_data, affine=aff)
    nib.save(seg_new, temp_dir)

    return temp_dir


if __name__ == "__main__":
    # Example:
    # python infer_single.py -cuda_device 0
    # nohup python -u infer_single.py -cuda_device 0 > tmp_single.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    
    # tgm16, essential
    """
    patient_dir = "/mnt/Drive2/lucas/datasets/data_GliODIL_essential/data_716"
    wmPath = os.path.join(patient_dir, "t1_wm.nii.gz")
    gmPath = os.path.join(patient_dir, "t1_gm.nii.gz")
    tumorSegmentationPath = os.path.join(patient_dir, "segm.nii.gz")
    savePath = os.path.join(patient_dir, "pet_ess")
    logfile = os.path.join(patient_dir, "tmp/gliodil.log")
    petPath = os.path.join(patient_dir, "FET.nii.gz")
    os.makedirs(savePath, exist_ok=True)
    tumorSegmentationPath_134 = tumorSegmentationPath
    """

    # tgm16, gbmdata
    patient_dir = "/mnt/Drive2/lucas/datasets/GLIODIL/tgm016/preop/preop/processed/"
    wmPath = os.path.join(patient_dir, "tissue_segmentation/wm_pbmap.nii.gz")
    gmPath = os.path.join(patient_dir, "tissue_segmentation/gm_pbmap")
    tumorSegmentationPath = os.path.join(patient_dir, "tumor_segmentation/tumor_seg.nii.gz")
    savePath = os.path.join(patient_dir, "growth_models/gliodil_test")
    logfile = os.path.join(savePath, "gliodil.log")
    os.makedirs(savePath, exist_ok=True)
    tumorSegmentationPath_134 = convert_tumorseg_labels(tumorSegmentationPath)

    # With PET
    #cmd = f'USEGPU=1 CUDA_VISIBLE_DEVICES={args.cuda_device} /home/home/lucas/projects/GliODIL/GliODIL.py --outdirectory "{savePath}" --optimizer adamn --postfix _pet__PDE1.0_ --lambda_pde_multiplier 1.0 --Nt 192 --Nx 48 --Ny 48 --Nz 48 --days 100 --history_every 1000 --report_every 1000 --epochs 9000 --plot_every 3000 --save_solution y --final_print y --code x --multigrid 1 --save_forward odil_res --save_forward2 full_trim_Gauss --initial_guess forward_character_dice_breaking --seg_path "{tumorSegmentationPath_134}" --wm_path "{wmPath}"  --gm_path "{gmPath}" --pet_path "{petPath}"'
    cmd = f'USEGPU=1 CUDA_VISIBLE_DEVICES={args.cuda_device} /home/home/lucas/projects/GliODIL/GliODIL.py --outdirectory "{savePath}" --optimizer adamn --postfix _pet__PDE1.0_ --lambda_pde_multiplier 1.0 --Nt 192 --Nx 48 --Ny 48 --Nz 48 --days 100 --history_every 1000 --report_every 1000 --epochs 9000 --plot_every 3000 --save_solution y --final_print y --code x --multigrid 1 --save_forward odil_res --save_forward2 full_trim_Gauss --initial_guess forward_character_dice_breaking --seg_path "{tumorSegmentationPath_134}" --wm_path "{wmPath}"  --gm_path "{gmPath}" --pet_path ""'
    print(cmd)
    
    
    os.system(cmd)
    print("Done.")
