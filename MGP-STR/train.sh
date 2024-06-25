CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  --master_port 29501 train.py --train_data data/training --valid_data data/eval --select_data / --batch_ratio 1 \
--Transformer mgp-str --TransformerModel=mgp_str_small_patch4_3_32_128 --imgH 32 --imgW 128 --manualSeed=226 \
--workers=0 --isrand_aug --scheduler --batch_size=100 --rgb --saved_path ./model_files --exp_name mgp_str_kor \
--valInterval 500 --num_iter 2000000 --lr 1 --character korean_dict.pkl
