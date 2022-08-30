offset_mode=quad
name=MTRdeblur_${offset_mode}
blur_direction=deblur    # reblur/deblur

python test.py \
       --name=$name \
       --offset_mode=${offset_mode} \
       --gpu_ids=1 \
       --no_crop \
       --blur_direction=${blur_direction} \
       --checkpoints_dir='pretrain_models'

python metrics.py --res_root="./exp_results/$name/test_latest/images"
