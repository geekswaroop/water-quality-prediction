# water-quality-prediction

This is an implementation of a novel deep learning model for predicting water quality.  

## Instructions to Run

```
python3 main.py -e <Number_of_epochs> -lr <learning_rate> -model <classifier_type> -tts <train_test_split_ratio>
```
- Classifier types are defined in ```model.py```, and should be one of 'classifier1', 'classifier2', 'classifier3' or 'classifier4'. Default is classifier1
- Number of epochs in training : Default is 500
- Learning rate during Stochastic Gradient Descent : Default is 0.1
- Train-test-split ratio : Fraction of the dataset that should be used for testing. Remaining will be used for training. Default is 0.45.
