#name=blur_quad_reg3
offset_method=quad
name= MTR_Gopro_${offset_method}

python train.py \
       --name=$name \
       --offset_method=${offset_method} \
       --gpu_ids=1 \
       --blur_direction=reblur \
       --dataset_mode=aligned \
       --niter_decay=800
