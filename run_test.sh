#name=blur_quad_reg3
offset_method=quad
name=syn_${offset_method}_400epoch
#name=Gopro_${offset_method}_blur_wotv
epoch=latest
python test.py \
       --name=$name \
       --offset_method=${offset_method} \
       --gpu_ids=1 \
       --no_crop \
       --blur_direction=reblur

python metrics.py --res_root="./exp_results/$name/test_${epoch}/images"
