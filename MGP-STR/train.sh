export WANDB_API_KEY="0a7cca3a906f5c34a06fe63623461725e2278ef3"
export WANDB_ENTITY="hero981001"
export TOKENIZERS_PARALLELISM="false"
ulimit -n 65535

nohup env OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0,1  TOKENIZERS_PARALLELISM=false \
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  --master_port 29510 train.py \
--train_data data/train/ --valid_data data/val/ \
--Transformer mgp-str --TransformerModel=mgp_str_base_patch4_3_32_128 --imgH 32 --imgW 128 --manualSeed=456 \
--workers 12 --scheduler --batch_size=50 --saved_path ./model_files \
--valInterval 20000 --num_iter 2000000 --lr 0.3 --character korean_dict_noCn.pkl \
--exp_name mgp_gray_base_max30_tokenizer_sensitive --batch_max_length 30 \
--input_channel 1 \
--PAD --isrand_aug --sensitive \
--select_data korquad_half-law_fromval-pul_data-realdata3_2_train-kowiki_fromval-random_half-korquad-BR_data-naverreview-fakedata_half_fromval-naverreview_half_fromval-law_half-fakedata_half-realdata1_2_train-address-banklmdb_4-fakedata-fakedata_wl_fromval-random-korquad_half_fromval-banklmdb_1-kowiki_wl_fromval-namuwiki_wl_fromval-banklmdb_5-bnk_lmdb_new-enwiki_half-namuwiki_half_fromval-kowiki_half-fakedata_wl-address_half_fromval-enwiki_wl-address_half-law-fakedata_fromval-realdata4_2_train-namuwiki_half-enwiki_wl_fromval-nat_data-random_fromval-namuwiki_wl-banklmdb_2-realdata2_2_train-namuwiki_fromval-kowiki_28-naverreview_half-korquad_fromval-kowiki_wl-kowiki-enwiki-namuwiki-kowiki_half_fromval-banklmdb_3-address_fromval-enwiki_half_fromval \
--batch_ratio 0.03955438-0.00290933-0.002995828-0.049985727-0.01209733-0.016957522-0.03955438-0.008514905-0.013391607-0.002040893-0.002672752-0.011617932-0.009640209-0.046808112-0.010901874-0.000200845-0.009640209-0.001296084-0.016957522-0.009859125-0.000200845-0.004038002-0.002623958-0.000200845-0.020417141-0.053480176-0.009887383-0.064604581-0.005207829-0.002255702-0.015274617-0.010901874-0.011617932-0.002040893-0.050517123-0.055145994-0.003817853-0.003036079-0.003591462-0.005809377-0.000200845-0.068278023-0.009887383-0.045179752-0.013391607-0.009859125-0.016179116-0.064604581-0.053480176-0.055145994-0.01209733-0.000200845-0.002255702-0.013381832 \
--saved_model /mnt/hdd1/sgh/MGP-STR/MGP-STR/model_files/mgp_gray_base_max30_tokenizer_sensitive-Seed456/iter_245000.pth \
--wandb \
> training_output.log 2>&1 &

# --saved_model ./model_files/Gray_large-Seed226/iter_385000.pth \
# --islearning_aug \
# --saved_model ./model_files/mgp_str_base_patch4_3_32_128-Seed226_aug-Seed226/iter_365000.pth \
# --exp_name mgp_str_base_patch4_3_32_128-Seed226_aug \
# -- \
# --batch_ratio 0.049977482-0.003675976-0.015285137-0.021426053-0.049977482-0.016920473-0.004537859-0.002578695-0.003377057-0.014679411-0.012180532-0.013774662-0.012180532-0.001637619-0.021426053-0.012457135-0.005102069-0.003315405-0.06757291-0.01249284-0.081628743-0.006580161-0.002850109-0.019299681-0.013774662-0.014679411-0.002578695-0.069677694-0.004823908-0.004537859-0.007340224-0.016908122-0.037424444-0.01249284-0.057085214-0.016920473-0.012457135-0.020442527-0.081628743-0.06757291-0.069677694-0.002850109-0.016908122 \
# --select_data korquad_half-law_fromval-kowiki_fromval-random_half-korquad-naverreview-random_half_fromval-fakedata_half_fromval-naverreview_half_fromval-law_half-fakedata_half-address-fakedata-fakedata_wl_fromval-random-korquad_half_fromval-kowiki_wl_fromval-namuwiki_wl_fromval-enwiki_half-namuwiki_half_fromval-kowiki_half-fakedata_wl-address_half_fromval-enwiki_wl-address_half-law-fakedata_fromval-namuwiki_half-enwiki_wl_fromval-random_fromval-namuwiki_wl-bnk_lmdb-namuwiki_fromval-kowiki_28-naverreview_half-korquad_fromval-kowiki_wl-kowiki-enwiki-namuwiki-kowiki_half_fromval-address_fromval-enwiki_half_fromval


# CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0,3  \
# python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  --master_port 29510 train.py \
# --train_data data/train/ --valid_data data/val/ \
# --Transformer mgp-str --TransformerModel=mgp_str_base_patch4_3_32_128 --imgH 32 --imgW 128 --manualSeed=456 \
# --workers 12 --scheduler --batch_size=80 --saved_path ./model_files \
# --valInterval 20000 --num_iter 2000000 --lr 0.5 --character korean_dict_noCn.pkl \
# --exp_name Test --batch_max_length 30 \
# --input_channel 1 \
# --PAD --isrand_aug --sensitive \
# --select_data korquad_half \
# --batch_ratio 1\



# CUDA_VISIBLE_DEVICES=2 \
# python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1  --master_port 29522 train.py \
# --train_data data/val/ --valid_data data/val/ \
# --Transformer mgp-str --TransformerModel=mgp_str_base_patch4_3_32_128 --imgH 32 --imgW 128 --manualSeed=456 \
# --workers 12 --scheduler --batch_size=1 --saved_path ./model_files \
# --valInterval 20000 --num_iter 2000000 --lr 0.5 --character korean_dict_noCn.pkl \
# --exp_name Test --batch_max_length 30 \
# --input_channel 1 \
# --PAD --sensitive \
# --select_data banklmdb_1 \
# --batch_ratio 1 \
# --saved_model /mnt/hdd1/sgh/MGP-STR/MGP-STR/model_files/mgp_gray_base_max30_tokenizer_sensitive-Seed456/iter_245000.pth \
