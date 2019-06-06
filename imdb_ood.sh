python -m text.run --batch-size 32 --z-dim 5 --model dwac \
    --dataset imdb --n-proto 64 \
    --device 0 --output-dir ./data/ood_stackoverflow/dwac

python -m text.run --batch-size 32 --z-dim 5 --model baseline \
    --dataset imdb --n-proto 64 \
    --device 0 --output-dir ./data/ood_stackoverflow/baseline

python -m text.run --batch-size 32 --z-dim 5 --model proto \
    --dataset imdb --n-proto 64 \
    --device 0 --output-dir ./data/ood_stackoverflow/proto_64

python -m text.run --batch-size 32 --z-dim 5 --model proto \
    --dataset imdb --n-proto 16 \
    --device 0 --output-dir ./data/ood_stackoverflow/proto_16

python -m text.run --batch-size 32 --z-dim 5 --model proto \
    --dataset imdb --n-proto 4 \
    --device 0 --output-dir ./data/ood_stackoverflow/proto_4

python -m text.run --batch-size 32 --z-dim 5 --model proto \
    --dataset imdb --n-proto 1 \
    --device 0 --output-dir ./data/ood_stackoverflow/proto_1
