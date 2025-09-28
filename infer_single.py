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

    rootdir = Path("/mnt/Drive2/lucas/datasets/GLIODIL")

    wmPath = str(rootdir / "t1_wm.nii.gz")
    gmPath = str(rootdir / "t1_gm.nii.gz")
    tumorSegmentationPath_134 = str(rootdir / "segm.nii.gz")
    savePath = str(rootdir / "preop/processed/growth_models/gliodil")
    logfile = str(rootdir / "preop/processed/growth_models/gliodil/gliodil.log")
    os.makedirs(savePath, exist_ok=True)

    cmd = f'USEGPU=1 CUDA_VISIBLE_DEVICES={args.cuda_device} /home/home/lucas/projects/dockerize/GliODIL/GliODIL.py --outdirectory "{savePath}" --optimizer adamn --lambda_pde_multiplier 1.0 --Nt 192 --Nx 48 --Ny 48 --Nz 48 --days 100 --history_every 1000 --report_every 1000 --epochs 9000 --plot_every 3000 --save_solution y --final_print y --multigrid 1 --save_forward odil_res --save_forward2 full_trim_Gauss --initial_guess forward_character_dice_breaking --seg_path "{tumorSegmentationPath_134}" --wm_path "{wmPath}"  --gm_path "{gmPath}" --pet_path ""'
    print(cmd)

    os.system(cmd)
    print("Done.")
