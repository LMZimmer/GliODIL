import os
import shutil
import argparse
import numpy as np
import nibabel as nib


def convert_tumorseg_labels(seg_dir):
    os.makedirs("tmp", exist_ok=True)
    temp_dir = "tmp/tumorseg_134.nii.gz"

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

    CUDA_DEVICE = "0"
    tumorSegmentationPath = "/mlcube_io0/Patient-00000/00000-tumorseg.nii.gz"
    wmPath = "/mlcube_io0/Patient-00000/00000-wm.nii.gz"
    gmPath = "/mlcube_io0/Patient-00000/00000-gm.nii.gz"
    csfPath = "/mlcube_io0/Patient-00000/00000-csf.nii.gz"
    savePath = "tmp"
    logfile = "tmp/gliodil.log"

    tumorSegmentationPath_134 = convert_tumorseg_labels(tumorSegmentationPath)

    # Run without PET
    cmd = f'USEGPU=1 CUDA_VISIBLE_DEVICES={CUDA_DEVICE} /app/GliODIL.py --outdirectory "{savePath}" --optimizer adamn --postfix _pet__PDE1.0_ --lambda_pde_multiplier 1.0 --Nt 192 --Nx 48 --Ny 48 --Nz 48 --days 100 --history_every 1000 --report_every 1000 --epochs 9000 --plot_every 3000 --save_solution y --final_print y --code x --multigrid 1 --save_forward odil_res --save_forward2 full_trim_Gauss --initial_guess forward_character_dice_breaking --seg_path "{tumorSegmentationPath_134}" --wm_path "{wmPath}"  --gm_path "{gmPath}" --pet_path ""'
    
    print(cmd)
    os.system(cmd)

    # Copy to mlcubeio1 and cleanup
    pred_file = os.path.join(savePath+"x_pet__PDE1.0_", "192_48_48_48_solution.nii")
    img = nib.load(pred_file)
    nib.save(img, "/mlcube_io1/00000.nii.gz")

    shutil.rmtree(savePath)
    shutil.rmtree(savePath+"x_pet__PDE1.0_")

    print("Done.")
