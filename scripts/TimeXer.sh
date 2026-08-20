export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_DISABLED=1
cd /home/zhao/fu/Time-Series-Library-main/
model_name=TimeXer2
pred_len=96
#for  pred_len in  192 336 720
# do
#
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/cite30MW/ \
  --data_path 3-6.csv \
  --model_id solar3_96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 1024 \
  --batch_size 16 \
  --itr 1 \
##
#  done
#
#
#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/cite30MW/ \
#  --data_path 1-6.csv \
#  --model_id solar1_96_192 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 192 \
#  --e_layers 3 \
#  --factor 3 \
#  --enc_in 6 \
#  --dec_in 6 \
#  --c_out 6 \
#  --des 'Exp' \
#  --d_model 128 \
#  --d_ff 1024 \
#  --batch_size 4 \
#  --itr 1

#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/cite30MW/ \
#  --data_path 2-6.csv \
#  --model_id solar2_96_336 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 336 \
#  --e_layers 1 \
#  --factor 3 \
#  --enc_in 6 \
#  --dec_in 6 \
#  --c_out 6 \
#  --des 'Exp' \
#  --d_model 256 \
#  --batch_size 4 \
#  --itr 1

#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/cite30MW/ \
#  --data_path 1-6.csv \
#  --model_id solar1_96_720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 720 \
#  --e_layers 1 \
#  --factor 3 \
#  --enc_in 6 \
#  --dec_in 6 \
#  --c_out 6 \
#  --des 'Exp' \
#  --d_model 128 \
#  --batch_size 4 \
#  --itr 1
