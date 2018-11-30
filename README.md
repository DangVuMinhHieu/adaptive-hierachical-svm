# adaptive-hierachical-svm

## Build model
#### Sử dụng file generate_db.py để sinh tập file train và test
Cú pháp: 
```
python generate_db.py src dst
```
Trong đó src là folder images của caltech-256, dst là folder sẽ chứa 2 file train.txt và test.txt chứa đường dẫn các file ảnh tương ứng (train.txt chiếm 70% tổng số ảnh, test.txt là 30%)

#### Sử dụng file features_extractor.py để extract các feature trong tập train và lưu xuống file trong folder features/vgg16_fc2
Cú pháp: 
```
python features_extractor.py src
```
Trong đó src là đường dẫn file train.txt

#### Train model
Cú pháp:
```
python adaptive_hierachical_svm.py src
```
Trong đó src là đường dẫn file train.txt

#### Test
Cú pháp
```
python test.py test_src
```
Trong đó test_src là đường dẫn đến feautures extract từ ảnh test (nên là features/vgg16_fc2/test/)
```
