name=VLNBERT-reasoner_v4p3

flag="--vlnbert prevalent
      --test_only 0
      --train auglistener
      --aug /egr/research-hlr/joslin/r2r/data/reason_aug1.json
      --features places365
      --maxAction 15
      --batchSize 8
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
