# HDA: COVID detection
Final project for the Human Data Analytics course. Topic: COVID detection from X-ray scans. Models employed: 
- a "simple" CNN model
- self-implemented CNN with attention.

Models also tested on the ```eurosat``` dataset because both COVID datasets

Files:
- proj.ipynb: the old project, deep CNN with attention
- base_model.py: definition of the baseline CNN model
- att_model.py: definition of the linear attention CNN model
- train_att_model.py, train_base_model.py: code for training the models.

Usage: ```python train_att_model.py -sh 256 256 -c 3 -b 32 -e 30 -p "./covid_data"```

```rb
  --input_shape INPUT_SHAPE INPUT_SHAPE, -sh INPUT_SHAPE INPUT_SHAPE //input shape: height, width
  --num_classes NUM_CLASSES, -c NUM_CLASSES //number of classes: int
  --batch_size BATCH_SIZE, -b BATCH_SIZE //batch size: int
  --epochs EPOCHS, -e EPOCHS //number of epochs: int
  --path PATH, -p PATH  //data path or "eurosat" for tfds eurosat dataset
```



Data: 
- https://data.mendeley.com/datasets/jctsfj2sfn/1
- https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database 
