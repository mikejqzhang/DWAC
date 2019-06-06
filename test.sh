python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class hibernate visual-studio magento wordpress \
    --output-dir ./data/06/hibernate_visual-studio_magento_wordpress \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class linq oracle hibernate qt \
    --output-dir ./data/06/linq_oracle_hibernate_qt \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class drupal bash qt wordpress \
    --output-dir ./data/06/drupal_bash_qt_wordpress \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class sharepoint apache spring matlab \
    --output-dir ./data/06/sharepoint_apache_spring_matlab \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class haskell osx sharepoint cocoa \
    --output-dir ./data/06/haskell_osx_sharepoint_cocoa \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class spring apache visual-studio scala \
    --output-dir ./data/06/spring_apache_visual-studio_scala \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class magento drupal visual-studio haskell \
    --output-dir ./data/06/magento_drupal_visual-studio_haskell \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class magento apache wordpress osx \
    --output-dir ./data/06/magento_apache_wordpress_osx \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class ajax linq bash qt \
    --output-dir ./data/06/ajax_linq_bash_qt \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model proto \
    --dataset stackoverflow --n-proto 64 \
    --ood-class cocoa qt matlab hibernate \
    --output-dir ./data/06/cocoa_qt_matlab_hibernate \
    --device 0


python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class hibernate visual-studio magento wordpress \
    --output-dir ./data/06/hibernate_visual-studio_magento_wordpress \
    --device 0
python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class linq oracle hibernate qt \
    --output-dir ./data/06/linq_oracle_hibernate_qt \
    --device 0
python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class drupal bash qt wordpress \
    --output-dir ./data/06/drupal_bash_qt_wordpress \
    --device 0
python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class sharepoint apache spring matlab \
    --output-dir ./data/06/sharepoint_apache_spring_matlab \
    --device 0
python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class haskell osx sharepoint cocoa \
    --output-dir ./data/06/haskell_osx_sharepoint_cocoa \
    --device 0
python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class spring apache visual-studio scala \
    --output-dir ./data/06/spring_apache_visual-studio_scala \
    --device 0
python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class magento drupal visual-studio haskell \
    --output-dir ./data/06/magento_drupal_visual-studio_haskell \
    --device 0
python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class magento apache wordpress osx \
    --output-dir ./data/06/magento_apache_wordpress_osx \
    --device 0
python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class ajax linq bash qt \
    --output-dir ./data/06/ajax_linq_bash_qt \
    --device 0
python -m text.run --batch-size 64 --z-dim 50 --model dwac \
    --dataset stackoverflow --n-proto 64 \
    --ood-class cocoa qt matlab hibernate \
    --output-dir ./data/06/cocoa_qt_matlab_hibernate \
    --device 0

python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class hibernate visual-studio magento wordpress \
    --output-dir ./data/06/hibernate_visual-studio_magento_wordpress \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class linq oracle hibernate qt \
    --output-dir ./data/06/linq_oracle_hibernate_qt \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class drupal bash qt wordpress \
    --output-dir ./data/06/drupal_bash_qt_wordpress \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class sharepoint apache spring matlab \
    --output-dir ./data/06/sharepoint_apache_spring_matlab \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class haskell osx sharepoint cocoa \
    --output-dir ./data/06/haskell_osx_sharepoint_cocoa \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class spring apache visual-studio scala \
    --output-dir ./data/06/spring_apache_visual-studio_scala \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class magento drupal visual-studio haskell \
    --output-dir ./data/06/magento_drupal_visual-studio_haskell \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class magento apache wordpress osx \
    --output-dir ./data/06/magento_apache_wordpress_osx \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class ajax linq bash qt \
    --output-dir ./data/06/ajax_linq_bash_qt \
    --device 0
python -m text.run --batch-size 32 --z-dim 50 --model baseline \
    --dataset stackoverflow --n-proto 64 \
    --ood-class cocoa qt matlab hibernate \
    --output-dir ./data/06/cocoa_qt_matlab_hibernate \
    --device 0
