# CUDA_VISIBLE_DEVICES=0 python test_final.py --eval_data ./data/val/enwiki --Transformer mgp-str  \
# --TransformerModel=mgp_str_base_patch4_3_32_128 \
# --model_dir ./model_files/mgp_gray_base_max30_tokenizer_sensitive-Seed456/best_accuracy.pth --batch_size 100 --PAD \
# --character ./korean_dict_noCn.pkl --batch_max_length 30 --workers 12 --input_channel 1



for iter in {385000..420000..5000}
do
    CUDA_VISIBLE_DEVICES=2 python test_final.py --eval_data ./data/train/bnk_lmdb_new --Transformer char-str  \
    --TransformerModel=char_str_large_patch8_1_32_224 \
    --model_dir ./model_files/char_large_max25-Seed456/iter_${iter}.pth --batch_size 100 --PAD \
    --character ./korean_dict_noCn.pkl --batch_max_length 25 --workers 12 --input_channel 1 --imgW 224
done