#name=blur_quad_reg3
offset_method=bilin
name=blur_${offset_method}_GT
epoch=latest
python train.py \
       --name=$name \
       --offset_method=${offset_method} \
       --gpu_ids=1 \
       --blur_direction=reblur \
       --dataset_mode=aligned_map \
       --niter_decay=800
# python metrics.py --res_root="./results/$name/test_${epoch}/images"
