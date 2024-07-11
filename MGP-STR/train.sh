CUDA_VISIBLE_DEVICES=1  python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1  --master_port 29501 train.py \
--train_data data/train --valid_data data/val --select_data random_half-korquad_half_fromval-address_fromval-address_half_fromval-enwiki_half_fromval-naverreview_half-kowiki_wl_fromval-namuwiki_half-enwiki_half-law_half-naverreview-law_half_fromval-cn_base_train-fakedata-namuwiki_wl_fromval-address-random-random_half_fromval-enwiki_wl_fromval-namuwiki-fakedata_half-law_fromval-korquad_fromval-kowiki_28-law-fakedata_fromval-fakedata_wl_fromval-korquad-namuwiki_half_fromval-fakedata_wl-kowiki_half-kowiki_wl-enwiki-namuwiki_wl-kowiki-random_fromval-address_half-enwiki_wl-cn_aug_train-naverreview_half_fromval-fakedata_half_fromval-korquad_half-kowiki_half_fromval-kowiki_fromval \
--batch_ratio 0.014717646-0.008556859-0.001957752-0.001957752-0.01161426-0.011622744-0.003504633-0.047861901-0.046416116-0.010083349-0.011622744-0.002525044-0.28517872-0.008366858-0.002277366-0.009461873-0.014717646-0.003117075-0.003313563-0.047861901-0.008366858-0.002525044-0.008556859-0.039212074-0.010083349-0.001771316-0.001124887-0.034329743-0.008581384-0.00451994-0.056071127-0.014042058-0.046416116-0.005042031-0.056071127-0.003117075-0.009461873-0.013257032-0.07129468-0.002319715-0.001771316-0.034329743-0.010499425-0.010499425 \
--Transformer mgp-str --TransformerModel=mgp_str_base_patch4_3_32_128 --imgH 32 --imgW 128 --manualSeed=226 \
--workers 24 --scheduler --batch_size=130 --rgb --saved_path ./model_files \
--valInterval 5000 --num_iter 2000000 --lr 1 --character korean_dict.pkl --saved_model ./model_files/mgp_str_base_patch4_3_32_128-Seed226/iter_130000.pth \
--exp_name mgp_str_base_patch4_3_32_128-Seed226_aug --isrand_aug


