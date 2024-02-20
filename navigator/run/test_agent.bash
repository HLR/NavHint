name=VLNBERT-best

flag="--vlnbert prevalent

      --submit 0
      --test_only 0

      --train validlistener
      --load snap/VLNBERT-reasoner_v4_best/state_dict/best_val_unseen
      --features places365
      --maxAction 15
      --batchSize 4
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=7 python r2r_src_reasoner/train.py $flag --name $name
