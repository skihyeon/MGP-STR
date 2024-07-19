OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=1,2,3  python3 -m torch.distributed.launch --nproc_per_node=3 --nnodes=1  --master_port 29502 train.py \
--train_data data/train --valid_data data/val --select_data korquad_half-law_fromval-kowiki_fromval-random_half-korquad-naverreview-random_half_fromval-fakedata_half_fromval-naverreview_half_fromval-law_half-fakedata_half-address-fakedata-fakedata_wl_fromval-random-korquad_half_fromval-kowiki_wl_fromval-namuwiki_wl_fromval-enwiki_half-namuwiki_half_fromval-kowiki_half-fakedata_wl-address_half_fromval-enwiki_wl-address_half-law-fakedata_fromval-namuwiki_half-enwiki_wl_fromval-random_fromval-namuwiki_wl-bnk_lmdb-namuwiki_fromval-kowiki_28-naverreview_half-korquad_fromval-kowiki_wl-kowiki-enwiki-namuwiki-kowiki_half_fromval-address_fromval-enwiki_half_fromval \
--batch_ratio 0.049977482-0.003675976-0.015285137-0.021426053-0.049977482-0.016920473-0.004537859-0.002578695-0.003377057-0.014679411-0.012180532-0.013774662-0.012180532-0.001637619-0.021426053-0.012457135-0.005102069-0.003315405-0.06757291-0.01249284-0.081628743-0.006580161-0.002850109-0.019299681-0.013774662-0.014679411-0.002578695-0.069677694-0.004823908-0.004537859-0.007340224-0.016908122-0.037424444-0.01249284-0.057085214-0.016920473-0.012457135-0.020442527-0.081628743-0.06757291-0.069677694-0.002850109-0.016908122 \
--Transformer mgp-str --TransformerModel=mgp_str_base_patch4_3_32_128 --imgH 32 --imgW 128 --manualSeed=226 \
--workers 12 --scheduler --batch_size=50 --saved_path ./model_files \
--valInterval 10000 --num_iter 2000000 --lr 0.5 --character korean_dict.pkl \
--saved_model ./model_files/mgp_str_base_patch4_3_32_128-Seed226_aug-Seed226/iter_365000.pth \
--exp_name mgp_str_base_patch4_3_32_128-Seed226_aug \
--rgb --PAD
