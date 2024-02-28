export CUDA_VISIBLE_DEVICES=0
#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
PYTHONPATH=$PWD:$PYTHONPATH python conflict.py \
	--dataset camvid \
    --arch network.deepv3.DeepWV3Plus \
    --inference_mode sliding \
    --scales 1.0 \
    --split test \
    --cv_split 0 \
    --dump_images \
    --ckpt_path ${2} \
    --snapshot ${1}

# bash scripts/eval_camvid.sh pretrained_models/camvid_best.pth results/dropoout_test
# bash scripts/eval_cityscapes_SEResNeXt50.sh pretrained_models/cityscapes_best.pth results
# bash scripts/eval_camvid.sh pretrained_models/camvid_best.pth test_idr
# bash scripts/eval_camvid.sh logs/camvid_ft/camv-network.deepv3_modify.DeepWV3Plus_dropout_apex_T_class_uniform_tile_720_crop_size_640_dataset_camvid_rlx_loss_local_rank_0_rlx_off_epoch_80_scale_min_0.8_PT_sbn/best_epoch_52_mean-iu_0.76629.pth results/dropout
# bash scripts/eval_camvid.sh pretrained_models/camvid_dropout.pth test_idr