# LSTM Handling Classifier

## Folder structure
```
lstm_project/
    data/
        dry_laps.csv       ← put your dry CSV here, rename it this
        wet_laps.csv       ← put your wet CSV here, rename it this
    dataset_lstm.py
    model_lstm.py
    train_lstm.py
    test_wet.py
```

## Setup
```
pip install torch pandas numpy
```

## Step 1 — Train on dry data
```
python train_lstm.py
```
This saves the best model to handling_lstm.pth

## Step 2 — Test on wet data
```
python test_wet.py
```
This loads the dry-trained model and runs it on wet data.
The accuracy drop between train validation and wet test is the result.
