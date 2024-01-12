# Classification Metrics.

1. Get code from https://github.com/KamilyaKharisova/simple_classifier
2. Get database using Sportsmanheight class. Get model confidence, that sportsman is basketball player, using classifier Classifier.
3. Change threshold for model confidence to classify sportsman sport (basketball or football). If confidence is higher or equal than  threshold  we define sportsman as basketball player, if  confidence is lower than  threshold  the algorithm we sportsman as football player.  For each threshold calculate TP, FP,F N, TN, accuracy, recall, precision, f1 score values. Example of threshold change is in table, that I send
4. Build precision-recall curve (precision y axis, recall x axis) in plotly. For each point calculate accuracy, f1 score. Add to hover_data accuracy, f1 score and threshold. Calculate area under curve. Add area value to the title of the plot
5. Bonus build ROC curve and calculate area under curve.

![image](https://github.com/ilyaskalimullinn/ml_hw3/assets/90423658/bb9e6bcf-97e8-4e6c-b663-a3ae9015c275)

![image](https://github.com/ilyaskalimullinn/ml_hw3/assets/90423658/7922bcc8-0d37-400c-9937-6fed9aa00085)
