# python train.py --lr 0.0005 --model_name "ATC_6M_LR=5e-4"
# python train.py --lr 0.0008 --model_name "ATC_6M_LR=8e-4"
# python train.py --lr 0.001 --model_name "ATC_6M_LR=1e-3"

# python train.py --lr 0.0008 --model_name "ATC_FINAL"

# python train.py --mode "complex" --reverse False --model_name "ATC_no_reverse"
# python train.py --mode "complex" --reverse True --model_name "ATC_reverse"
python train.py --mode "complex" --lr 0.0008 --model_name "ATC_no_reverse" 
