# Tweet sentiment CLI

This CLI loads the selected classical model exported by `Sentiment-Analysis.ipynb`. The notebook exports the fitted pipeline and selected preprocessing setting to:

```text
Project/ml-cli/sentiment_model.joblib
```

`preprocessing.py` shares the same tweet cleanup and tokenization functions used to train the sklearn pipeline. It uses the selected stopword and lemmatization settings stored with the exported model.

## Prepare the model

Run `Sentiment-Analysis.ipynb` with the project `.venv313` kernel through Section 8.5, after the final refit. The export cell writes `sentiment_model.joblib` here. The CLI requires scikit-learn, joblib, Emoji, and NLTK in that environment.

## Use it

From the workspace root, pass a tweet for a single prediction:

```powershell
.\.venv313\Scripts\python.exe Project\ml-cli\app.py "I am so happy about this!"
```

Run without a tweet to start an interactive session. Enter multiple tweets, use `/help` for commands, and `/exit` to quit:

```powershell
.\.venv313\Scripts\python.exe Project\ml-cli\app.py
```
