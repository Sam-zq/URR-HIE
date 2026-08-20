export CUDA_VISIBLE_DEVICES=0
model_name=URR-HIE

for  pred_len in  96 192  336 720
 do



python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/cite30MW/ \
  --data_path 3-6.csv \
  --model_id solar3_norm_a=0.2_96-$pred_len\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --batch_size 16 \
  --activation gelu \
  --dropout 0.1 \
  --learning_rate 0.0005 \
  --lradj type1 \
  --train_epochs 10 \
  --patience 3 \

  done

