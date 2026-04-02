import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# load dataset
df = pd.read_csv("../dataset/phishing.csv")

# remove Index column if present
if "Index" in df.columns:
    df = df.drop(columns=["Index"])

# separate features and label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# convert labels
# dataset uses:
# -1 = safe
#  1 = phishing

y = y.map({-1: 0, 1: 1})

print("Unique labels after conversion:", y.unique())

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric="logloss",
)

model.fit(X_train, y_train)

# evaluate
pred = model.predict(X_test)

print(classification_report(y_test, pred))

# save model
joblib.dump(model, "phishing_model.pkl")

print("Model saved successfully")
