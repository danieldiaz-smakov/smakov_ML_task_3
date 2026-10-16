# Model card: term deposit call ranking

## Purpose

The bank calls clients to sell a term deposit, but the call centre can only call 10% of a list in one campaign wave. The model gives each client a score before any call is made, and the 10% of clients with the highest scores get a call. It's meant only for ranking clients inside one list, not for deciding anything else about a client.

## Data

UCI Bank Marketing dataset (bank-additional-full.csv, Moro, Rita & Cortez, 2014, CC BY 4.0), calls from May 2008 to November 2010. It's a teaching dataset, so the results shouldn't be read as facts about real banks or clients. train.csv has 32940 rows with the answer y (11.3% yes), test.csv has 8236 rows without it.

duration (the length of the call) is in train.csv but not in test.csv. It's only known after the call has ended, so it can't be used for scoring before the call, and I dropped it. id is dropped too. There are no NaN values, but several columns have "unknown" (default 20.9% of rows, education 4.3%, housing and loan 2.4%), and pdays = 999 (no previous contact recorded) appears in 96.2% of rows.

## Pipeline

Everything is one sklearn Pipeline that takes the raw test.csv columns (without id):
- categorical columns: one-hot encoding, "unknown" kept as its own category
- pdays: 999 replaced by the median of the real values, plus a 0/1 flag column for 999, then scaled
- other numeric columns: StandardScaler
- model: random forest, 300 trees, max_features = 0.3, min_samples_leaf = 20, no class weights, random_state = 3194

The final model is trained on all of train.csv with scikit-learn 1.7.2 and saved as model.joblib (9.4 MB).

## Metric and model choice

I kept 20% of train.csv (6588 rows) as a hold-out and did all comparisons with stratified 5-fold CV on the remaining 26352 rows. The main metric was average precision (AP), because with the 10% rule only the top of the ranking matters. I also tracked ROC-AUC. Accuracy is useless here: always saying "no" already gives 88.7%.

I compared logistic regression, linear SVM, KNN, a decision tree and a random forest, each tuned with grid search on the same folds. The random forest was the best (AP 0.482 ± 0.011) and beat logistic regression (0.458 ± 0.011) in all 5 folds. Logistic regression, linear SVM and the tree were within noise of each other (0.453-0.458), and KNN was a bit lower (0.442). Class weights didn't help any model, and the alternatives I tried for "unknown" and pdays = 999 gave the same results within noise.

## Results on the hold-out

ROC-AUC 0.788, AP 0.443 (a bit lower than in CV, which was 0.805 and 0.482).

With the 10% rule (659 calls out of 6588 clients), 326 called clients say yes: precision 49.5%, recall 43.9%, lift 4.4x over calling at random, which would give about 74 yes.

## Limits and how it can fail

The validation split is random, so train and hold-out come from the same time periods. In real use the model would be trained on the past and used on the future, so I expect the real numbers to be lower.

The strongest features are economic indicators (nr.employed, euribor3m, emp.var.rate). They describe the period, not the client. If the economy changes, or these values go outside the range seen in 2008-2010, the ranking can get much worse.

The data is from 2008-2010. Clients, channels (for example cellular vs telephone) and the offer itself can change, and the model wouldn't know.

Problems with input data can fail silently. A new category value (a new job type, for example) is just ignored by the encoder without an error. If pdays starts coding "no previous contact" differently than 999 (for example as -1), the model will treat it as a real number of days.

The scores are good for ranking but shouldn't be read as exact probabilities.

If the model is retrained only on clients it chose to call, the training data becomes biased toward the kind of clients it already likes.

## What to monitor after launch

- the share of yes among called clients in each wave, compared to about 50% on the hold-out
- the economic indicators and other inputs compared to the training data, especially values outside the training range
- the share of "unknown" values and any new category values
- retraining from time to time, checking the new model on the most recent campaign instead of a random split