name=test_r4r

flag="--vlnbert prevalent

      --submit 0
      --test_only 0

      --train validlistener
      --load /localscratch/zhan1624/VLN-interactive/snap/VLNBERT-r4r_reasoner/state_dict/best_val_unseen
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
CUDA_VISIBLE_DEVICES=6 python r4r_src_reasoner_v4/train.py $flag --name $name
