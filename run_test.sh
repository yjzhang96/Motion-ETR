#name=blur_quad_reg3
offset_method=quad
name=Gopro_motion_${offset_method}
#name=Gopro_${offset_method}_blur_wotv
epoch=latest
python test.py \
       --name=$name \
       --offset_method=${offset_method} \
       --gpu_ids=1 \
       --no_crop \
       --blur_direction=reblur \
       --checkpoints_dir='pretrain_models'

python metrics.py --res_root="./exp_results/$name/test_${epoch}/images"
