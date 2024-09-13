export WANDB_API_KEY="0a7cca3a906f5c34a06fe63623461725e2278ef3"
export WANDB_ENTITY="hero981001"
export TOKENIZERS_PARALLELISM="false"
ulimit -n 65535

nohup env OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0,1  \
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  --master_port 29511 train.py \
--train_data data/train/ --valid_data data/val/ \
--Transformer char-str --TransformerModel=char_str_base_patch4_3_32_128 --imgH 32 --imgW 128 --manualSeed=456 \
--workers 12 --scheduler --batch_size=80 --saved_path ./model_files \
--valInterval 20000 --num_iter 2000000 --lr 1 --character korean_dict_noCn.pkl \
--exp_name char_max25_CTC --batch_max_length 25 \
--input_channel 1 \
--PAD --isrand_aug --sensitive \
--select_data korquad_half-law_fromval-pul_data-realdata3_2_train-kowiki_fromval-random_half-korquad-BR_data-naverreview-fakedata_half_fromval-naverreview_half_fromval-law_half-fakedata_half-realdata1_2_train-address-banklmdb_4-fakedata-fakedata_wl_fromval-random-korquad_half_fromval-banklmdb_1-kowiki_wl_fromval-namuwiki_wl_fromval-banklmdb_5-bnk_lmdb_new-enwiki_half-namuwiki_half_fromval-kowiki_half-fakedata_wl-address_half_fromval-enwiki_wl-address_half-law-fakedata_fromval-realdata4_2_train-namuwiki_half-enwiki_wl_fromval-nat_data-random_fromval-namuwiki_wl-banklmdb_2-realdata2_2_train-namuwiki_fromval-kowiki_28-naverreview_half-korquad_fromval-kowiki_wl-kowiki-enwiki-namuwiki-kowiki_half_fromval-banklmdb_3-address_fromval-enwiki_half_fromval \
--batch_ratio 0.03955438-0.00290933-0.002995828-0.049985727-0.01209733-0.016957522-0.03955438-0.008514905-0.013391607-0.002040893-0.002672752-0.011617932-0.009640209-0.046808112-0.010901874-0.000200845-0.009640209-0.001296084-0.016957522-0.009859125-0.000200845-0.004038002-0.002623958-0.000200845-0.020417141-0.053480176-0.009887383-0.064604581-0.005207829-0.002255702-0.015274617-0.010901874-0.011617932-0.002040893-0.050517123-0.055145994-0.003817853-0.003036079-0.003591462-0.005809377-0.000200845-0.068278023-0.009887383-0.045179752-0.013391607-0.009859125-0.016179116-0.064604581-0.053480176-0.055145994-0.01209733-0.000200845-0.002255702-0.013381832 \
--wandb \
> training_output.log 2>&1 &

# nohup env OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=2,3  TOKENIZERS_PARALLELISM=false \
# python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  --master_port 29511 train.py \
# --train_data data/train/ --valid_data data/val/ \
# --Transformer char-str --TransformerModel=char_str_base_patch4_3_32_128 --imgH 32 --imgW 128 --manualSeed=456 \
# --workers 12 --scheduler --batch_size=80 --saved_path ./model_files \
# --valInterval 20000 --num_iter 2000000 --lr 1 --character korean_dict_noCn.pkl \
# --exp_name char_max25_tl_bd --batch_max_length 25 \
# --input_channel 1 \
# --PAD --isrand_aug --sensitive \
# --select_data korquad_half-law_fromval-pul_data-realdata3_2_train-kowiki_fromval-random_half-korquad-BR_data-naverreview-fakedata_half_fromval-naverreview_half_fromval-law_half-fakedata_half-realdata1_2_train-address-banklmdb_4-fakedata-fakedata_wl_fromval-random-korquad_half_fromval-banklmdb_1-kowiki_wl_fromval-namuwiki_wl_fromval-banklmdb_5-bnk_lmdb_new-enwiki_half-namuwiki_half_fromval-kowiki_half-fakedata_wl-address_half_fromval-enwiki_wl-address_half-law-fakedata_fromval-realdata4_2_train-namuwiki_half-enwiki_wl_fromval-nat_data-random_fromval-namuwiki_wl-banklmdb_2-realdata2_2_train-namuwiki_fromval-kowiki_28-naverreview_half-korquad_fromval-kowiki_wl-kowiki-enwiki-namuwiki-kowiki_half_fromval-banklmdb_3-address_fromval-enwiki_half_fromval \
# --batch_ratio 0.03955438-0.00290933-0.002995828-0.049985727-0.01209733-0.016957522-0.03955438-0.008514905-0.013391607-0.002040893-0.002672752-0.011617932-0.009640209-0.046808112-0.010901874-0.000200845-0.009640209-0.001296084-0.016957522-0.009859125-0.000200845-0.004038002-0.002623958-0.000200845-0.020417141-0.053480176-0.009887383-0.064604581-0.005207829-0.002255702-0.015274617-0.010901874-0.011617932-0.002040893-0.050517123-0.055145994-0.003817853-0.003036079-0.003591462-0.005809377-0.000200845-0.068278023-0.009887383-0.045179752-0.013391607-0.009859125-0.016179116-0.064604581-0.053480176-0.055145994-0.01209733-0.000200845-0.002255702-0.013381832 \
# --wandb \
# > training_output2.log 2>&1 &

# nohup env OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0,1,2  TOKENIZERS_PARALLELISM=false \
# python3 -m torch.distributed.launch --nproc_per_node=3 --nnodes=1  --master_port 29510 train.py \
# --train_data data/train/ --valid_data data/val/ \
# --Transformer char-str --TransformerModel=char_str_large_patch8_1_32_224 --imgH 32 --imgW 224 --manualSeed=456 \
# --workers 12 --scheduler --batch_size=50 --saved_path ./model_files \
# --valInterval 20000 --num_iter 2000000 --lr 1 --character korean_dict_noCn.pkl \
# --exp_name char_large_max25 --batch_max_length 25 \
# --input_channel 1 \
# --PAD --issemantic_aug --sensitive \
# --select_data korquad_half-law_fromval-pul_data-realdata3_2_train-kowiki_fromval-random_half-korquad-BR_data-naverreview-fakedata_half_fromval-naverreview_half_fromval-law_half-fakedata_half-realdata1_2_train-address-banklmdb_4-fakedata-fakedata_wl_fromval-random-korquad_half_fromval-banklmdb_1-kowiki_wl_fromval-namuwiki_wl_fromval-banklmdb_5-bnk_lmdb_new-enwiki_half-namuwiki_half_fromval-kowiki_half-fakedata_wl-address_half_fromval-enwiki_wl-address_half-law-fakedata_fromval-realdata4_2_train-namuwiki_half-enwiki_wl_fromval-nat_data-random_fromval-namuwiki_wl-banklmdb_2-realdata2_2_train-namuwiki_fromval-kowiki_28-naverreview_half-korquad_fromval-kowiki_wl-kowiki-enwiki-namuwiki-kowiki_half_fromval-banklmdb_3-address_fromval-enwiki_half_fromval \
# --batch_ratio 0.03955438-0.00290933-0.002995828-0.049985727-0.01209733-0.016957522-0.03955438-0.008514905-0.013391607-0.002040893-0.002672752-0.011617932-0.009640209-0.046808112-0.010901874-0.000200845-0.009640209-0.001296084-0.016957522-0.009859125-0.000200845-0.004038002-0.002623958-0.000200845-0.020417141-0.053480176-0.009887383-0.064604581-0.005207829-0.002255702-0.015274617-0.010901874-0.011617932-0.002040893-0.050517123-0.055145994-0.003817853-0.003036079-0.003591462-0.005809377-0.000200845-0.068278023-0.009887383-0.045179752-0.013391607-0.009859125-0.016179116-0.064604581-0.053480176-0.055145994-0.01209733-0.000200845-0.002255702-0.013381832 \
# --wandb \
# > training_output.log 2>&1 &


# OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0,1,2  TOKENIZERS_PARALLELISM=false \
# python3 -m torch.distributed.launch --nproc_per_node=3 --nnodes=1  --master_port 29510 train.py \
# --train_data data/train/ --valid_data data/val/ \
# --Transformer char-str --TransformerModel=char_str_large_patch8_1_32_224 --imgH 32 --imgW 224 --manualSeed=456 \
# --workers 12 --scheduler --batch_size=50 --saved_path ./model_files \
# --valInterval 20000 --num_iter 2000000 --lr 0.5 --character korean_dict_noCn.pkl \
# --exp_name char_large_max25_onlybnk --batch_max_length 25 \
# --input_channel 1 \
# --PAD --sensitive \
# --select_data  bnk_lmdb_new \
# --batch_ratio 1 \
# --wandb \
# --saved_model /mnt/hdd1/sgh/MGP-STR/MGP-STR/model_files/char_large_max25-Seed456/iter_395000.pth \

