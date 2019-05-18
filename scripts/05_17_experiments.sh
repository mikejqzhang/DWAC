# python -m text.run --batch-size 32 --max-epochs 0 --device 0 --model proto --dataset imdb --n-proto 4 --output-dir ./data/tmp

# python -m text.run --batch-size 32 --device 0 --model proto --dataset imdb --n-proto 4 --output-dir ./data/05_17/imdb_proto_4
python -m text.run --batch-size 32 --device 0 --model proto --dataset imdb --n-proto 64 --output-dir ./data/05_17/imdb_proto_64
# python -m text.run --batch-size 32 --device 0 --model proto --dataset imdb --n-proto 256 --output-dir ./data/05_17/imdb_proto_256

# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 4 --output-dir ./data/05_17/stackoverflow_proto_4_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 64 --output-dir ./data/05_17/stackoverflow_proto_64_zdim_50
# python -m text.run --batch-size 32 --z-dim 30 --device 0 --model proto --dataset stackoverflow --n-proto 4 --output-dir ./data/05_17/stackoverflow_proto_4_zdim_30
# python -m text.run --batch-size 32 --z-dim 30 --device 0 --model proto --dataset stackoverflow --n-proto 64 --output-dir ./data/05_17/stackoverflow_proto_64_zdim_30


# python -m text.run --batch-size 32 --z-dim 5 --device 0 --model dwac --dataset imdb --n-proto 1 --output-dir ./data/temp/imdb_dwac
# python -m text.run --batch-size 32 --z-dim 5 --device 0 --model baseline --dataset imdb --n-proto 1 --output-dir ./data/temp/imdb_baseline

# python -m text.run --batch-size 128 --z-dim 50 --device 0 --model dwac --dataset stackoverflow --n-proto 1 --output-dir ./data/temp/stackoverflow_dwac_zdim_50_bs_64
# python -m text.run --lr 0.0001 --batch-size 64 --z-dim 50 --device 0 --model dwac --dataset stackoverflow --n-proto 1 --output-dir ./data/temp/stackoverflow_dwac_zdim_50_lr_1e-4_bs_64
# python -m text.run --lr 0.0001 --batch-size 32 --z-dim 50 --device 1 --model dwac --dataset stackoverflow --n-proto 1 --output-dir ./data/temp/stackoverflow_dwac_zdim_50_lr_1e-4
# python -m text.run --batch-size 32 --z-dim 50 --device 1 --model baseline --dataset stackoverflow --n-proto 1 --output-dir ./data/temp/stackoverflow_baseline

# python -m text.run --batch-size 32 --device 0 --model proto --dataset imdb --n-proto 1 --output-dir ./data/temp/imdb_proto_1
# python -m text.run --batch-size 32 --device 0 --model proto --dataset imdb --n-proto 16 --output-dir ./data/temp/imdb_proto_16
# python -m text.run --batch-size 32 --device 0 --model proto --dataset imdb --n-proto 64 --output-dir ./data/temp/imdb_proto_64
# python -m text.run --batch-size 32 --device 0 --model proto --dataset imdb --n-proto 256 --output-dir ./data/temp/imdb_proto_256
# python -m text.run --batch-size 32 --device 0 --model proto --dataset imdb --n-proto 1024 --output-dir ./data/temp/imdb_proto_1024

# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 1 --output-dir ./data/temp/stackoverflow_proto_1_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 4 --output-dir ./data/temp/stackoverflow_proto_4_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 16 --output-dir ./data/temp/stackoverflow_proto_16_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 64 --output-dir ./data/temp/stackoverflow_proto_64_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 256 --output-dir ./data/temp/stackoverflow_proto_256_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 1024 --output-dir ./data/temp/stackoverflow_proto_1024_zdim_50

# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset amazon --n-proto 1 --output-dir ./data/temp/amazon_proto_1_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset amazon --n-proto 4 --output-dir ./data/temp/amazon_proto_4_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset amazon --n-proto 16 --output-dir ./data/temp/amazon_proto_16_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset amazon --n-proto 64 --output-dir ./data/temp/amazon_proto_64_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset amazon --n-proto 256 --output-dir ./data/temp/amazon_proto_256_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset amazon --n-proto 1024 --output-dir ./data/temp/amazon_proto_1024_zdim_50
# 
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset subjectivity --n-proto 1 --output-dir ./data/temp/subjectivity_proto_1_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset subjectivity --n-proto 4 --output-dir ./data/temp/subjectivity_proto_4_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset subjectivity --n-proto 16 --output-dir ./data/temp/subjectivity_proto_16_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset subjectivity --n-proto 64 --output-dir ./data/temp/subjectivity_proto_64_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset subjectivity --n-proto 256 --output-dir ./data/temp/subjectivity_proto_256_zdim_50
# python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto --dataset subjectivity --n-proto 1024 --output-dir ./data/temp/subjectivity_proto_1024_zdim_50

# python -m text.run --batch-size 64 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 1 --output-dir ./data/temp/stackoverflow_proto_1_zdim_50_bs_64
# python -m text.run --batch-size 64 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 4 --output-dir ./data/temp/stackoverflow_proto_4_zdim_50_bs_64
# python -m text.run --batch-size 64 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 16 --output-dir ./data/temp/stackoverflow_proto_16_zdim_50_bs_64
# python -m text.run --batch-size 64 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 64 --output-dir ./data/temp/stackoverflow_proto_64_zdim_50_bs_64
# python -m text.run --batch-size 64 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 256 --output-dir ./data/temp/stackoverflow_proto_256_zdim_50_bs_64
# python -m text.run --batch-size 64 --z-dim 50 --device 0 --model proto --dataset stackoverflow --n-proto 1024 --output-dir ./data/temp/stackoverflow_proto_1024_zdim_50_bs_64
