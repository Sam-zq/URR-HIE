export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
cd /home/zhao/fu/Time-Series-Library-main/
model_name=n3
pred_len=96
seq_len=96

##
#for  pred_len in    336 720
# do

#done

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
#  --inverse \

#
#  done

#python -u run.py \
#  --task_name long_term_forecast \
#  --is_training 1 \
#  --root_path ./dataset/weather/ \
#  --data_path weather.csv \
#  --model_id weather_96_$pred_len \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len $pred_len \
#  --e_layers 3 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 21 \
#  --dec_in 21 \
#  --c_out 21 \
#  --des 'Exp' \
#  --itr 1 \
#  --d_model 256 \
#  --d_ff 1024 \
#  --n_heads 8 \
#  --batch_size 128 \
#  --activation gelu \
#  --dropout 0.1 \
#  --learning_rate 0.0001 \
#  --lradj type1 \
#  --train_epochs 10 \
#  --patience 3 \

# done
