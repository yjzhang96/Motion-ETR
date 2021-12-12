offset_method=quad
name=MTR_Gopro_${offset_method}
blur_direction=reblur    # reblur/deblur

python test.py \
       --name=$name \
       --offset_method=${offset_method} \
       --gpu_ids=1 \
       --no_crop \
       --blur_direction=${blur_direction} \
       --checkpoints_dir='pretrain_models'

python metrics.py --res_root="./exp_results/$name/test_latest/images"
