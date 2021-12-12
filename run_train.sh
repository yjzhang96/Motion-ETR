offset_mode=quad   # quad/lin/bilin
name=MTR_Gopro_${offset_mode}
blur_direction=reblur    # reblur/deblur

python train.py \
       --name=$name \
       --offset_mode=${offset_mode} \
       --gpu_ids=1 \
       --blur_direction=${blur_direction} \
       --dataset_mode=aligned \
       --niter_decay=800
