# Results

1. Dataset: 5_95
   model: LSTM 1
   Best validation accuracy: 14%
   Best testing accuracy: 15%
2. Dataset: 14_70
   model: LSTM 1
   Best validation accuracy: 18%
   Best testing accuracy: 19%
3. Dataset: 55_95
   model: LSTM 1
   Best validation accuracy: 22%
   Best testing accuracy: 22%
4. Dataset: 5_95
   model: LSTM 4
   Best validation accuracy: 28%
   Best testing accuracy: 28%
5. Dataset: raw
   model: LSTM 4
   Training accuracy: 83%
   Best validation accuracy: 73%
   Best testing accuracy: 72%
6. Dataset: raw
   model: LSTM 1
   Training accuracy: 39%
   Best validation accuracy: 29%
   Best testing accuracy: 30% 
7. Dataset: raw
   model: AttnSleep
   Training accuracy: 90%
   Best validation accuracy: 87.1%
   Best testing accuracy: 86.4% 

1. Compute the visual representation of all images
2. Train and compute the brain representation of each image (average brain representations for each signal)
3. Train the regressor to turn visual representations to their corresponding brain representation
4. Train the classifier to get the brain representation of the input signal and the brain representation of the input image and output one of the 40 classes
5. Make an abilation study