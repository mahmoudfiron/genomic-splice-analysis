import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# טוענים את הקובץ
df = pd.read_csv("S2.CSV")

# בודקים את מספר השורות והעמודות
print("Shape of dataframe:", df.shape)

# תצוגה של 5 השורות הראשונות לבדיקה
df.head()

# ------------------ סעיף 2 ------------------

# סעיף 2: חישוב ערכים חסרים (0) ו-missing rate עם דיוק
columns_to_check = [
    'PWM_ref', 'MES_ref', 'NNSplice_ref', 'HSF_ref',
    'GeneSplicer_ref', 'GENSCAN_ref', 'NetGene2_ref', 'SplicePredictor_ref'
]

print(f"{'Tool':<25} {'No. of missing':<20} {'Missing rate'}")
for col in columns_to_check:
    missing_count = (df[col] == 0).sum()
    missing_rate = (missing_count / df.shape[0])
    print("{:<25} {:<20} {:.3f}".format(col, missing_count, missing_rate))

# ------------------ סעיף 3 ------------------

# סעיף 3 – סינון שורות ללא missing בארבע שיטות עם missing rate < 0.05
top4_columns = ['PWM_ref', 'MES_ref', 'NNSplice_ref', 'HSF_ref']

# ניצור מסכה בוליאנית עבור כל עמודה בנפרד ונאחד ביניהן (AND)
non_missing_mask = (df[top4_columns[0]] != 0)
for col in top4_columns[1:]:
    non_missing_mask &= (df[col] != 0)

# אוסף האינדקסים של השורות שמתאימות (שבהן כל 4 הערכים שונים מ־0)
valid_indices = df[non_missing_mask].index

# שמירה לקובץ טקסט
with open("non_missing_top_4.txt", "w") as f:
    for idx in valid_indices:
        f.write(str(idx) + "\n")

# ------------------ סעיף 4 ------------------

# שלב א – טעינת האינדקסים התקינים מתוך סעיף 3
with open("non_missing_top_4.txt", "r") as f:
    good_indices = [int(line.strip()) for line in f.readlines()]

# סינון הדאטה רק לשורות התקינות
df_clean = df.loc[good_indices]

# שלב ב – חלוקה ל־train ו־test לפי מיקום (90/10)
split_index = int(len(df_clean) * 0.9)
df_train = df_clean.iloc[:split_index]
df_test = df_clean.iloc[split_index:]

# בדיקת תקינות החלוקה
assert len(df_clean) == len(df_train) + len(df_test)


# שלב ג – פונקציה לחישוב ופירוט positive / negative
def print_label_stats(name, data, label_col='Group'):
    total = len(data)
    pos_count = (data[label_col] == 'Positive').sum()
    neg_count = (data[label_col] == 'Negative').sum()
    pos_rate = (pos_count / total) * 100
    neg_rate = (neg_count / total) * 100

    print(f"\n{name} set:")
    print(f"Total samples: {total}")
    print(f"Positive: {pos_count} ({pos_rate:.2f}%)")
    print(f"Negative: {neg_count} ({neg_rate:.2f}%)")


# חישוב לכל קבוצה
print_label_stats("Full (cleaned)", df_clean)
print_label_stats("Train", df_train)
print_label_stats("Test", df_test)

# ------------------ סעיף 5 ------------------

# שלב 1 – סינון הדאטה לפי האינדקסים התקינים (non-missing מ־סעיף 3)
with open("non_missing_top_4.txt", "r") as f:
    good_indices = [int(line.strip()) for line in f.readlines()]

df_clean = df.loc[good_indices]

# שלב 2 – הפרדה לקבוצות positive ו-negative לפי העמודה 'Group'
df_pos = df_clean[df_clean['Group'] == 'Positive']
df_neg = df_clean[df_clean['Group'] == 'Negative']

# שלב 3 – חישוב גודל 90% לכל קבוצה
pos_split = int(len(df_pos) * 0.9)
neg_split = int(len(df_neg) * 0.9)

# שלב 4 – יצירת סטים train ו-test תוך שמירה על יחס דומה
df_train = pd.concat([
    df_pos.iloc[:pos_split],
    df_neg.iloc[:neg_split]
])

df_test = pd.concat([
    df_pos.iloc[pos_split:],
    df_neg.iloc[neg_split:]
])

# שלב 5 – בדיקת assert שהחלוקה מכסה את כל הדאטה
assert len(df_clean) == len(df_train) + len(df_test)


# שלב 6 – הדפסת נתוני יחס positive/negative בכל קבוצה
def print_label_stats(name, data, label_col='Group'):
    total = len(data)
    pos_count = (data[label_col] == 'Positive').sum()
    neg_count = (data[label_col] == 'Negative').sum()
    pos_rate = (pos_count / total) * 100
    neg_rate = (neg_count / total) * 100

    print(f"\n{name} set:")
    print(f"Total samples: {total}")
    print(f"Positive: {pos_count} ({pos_rate:.2f}%)")
    print(f"Negative: {neg_count} ({neg_rate:.2f}%)")


# הדפסת סיכום לכל אחת מהקבוצות
print_label_stats("Full (cleaned)", df_clean)
print_label_stats("Train", df_train)
print_label_stats("Test", df_test)

# ------------------ סעיף 6 ------------------
# סעיף 6 - א

# המרה של Group ל־0/1: Positive = 1, Negative = 0
y_train = (df_train['Group'] == 'Positive').astype(int)
y_test = (df_test['Group'] == 'Positive').astype(int)

# 1. טיפול במקרה של חילוק ב-0: נחליף אפסים ב-NaN
df_train['PWM_alt'] = df_train['PWM_alt'].replace(0, np.nan)
df_test['PWM_alt'] = df_test['PWM_alt'].replace(0, np.nan)

# 2. יצירת עמודת PWM_ratio בצורה בטוחה
df_train['PWM_ratio'] = df_train['PWM_ref'] / df_train['PWM_alt']
df_test['PWM_ratio'] = df_test['PWM_ref'] / df_test['PWM_alt']

# 3. סיווג לפי סף 1: אם ratio > 1 ⇒ Positive
train_pred_a = (df_train['PWM_ratio'] > 1).astype(int)
test_pred_a = (df_test['PWM_ratio'] > 1).astype(int)

# 4. הדפסת confusion matrix לפי סף 1
print("Confusion Matrix - Training Set (Threshold = 1):")
print(confusion_matrix(y_train, train_pred_a))

print("\nConfusion Matrix - Test Set (Threshold = 1):")
print(confusion_matrix(y_test, test_pred_a))


# סעיף 6 - ב

# 1. סינון רק השורות עם PWM_ratio תקין (לא NaN)
train_valid = df_train.dropna(subset=['PWM_ratio'])
y_train_valid = (train_valid['Group'] == 'Positive').astype(int)

# 2. חישוב ROC
fpr, tpr, thresholds = roc_curve(y_train_valid, train_valid['PWM_ratio'])

# 3. מציאת הסף שבו TPR הכי גבוה כאשר FPR ≤ 0.1
best_tpr = 0
best_thresh = 0

for i in range(len(thresholds)):
    if fpr[i] <= 0.1 and tpr[i] > best_tpr:
        best_tpr = tpr[i]
        best_thresh = thresholds[i]

print(
    f"\nBest threshold with FPR <= 0.1: {best_thresh:.4f} "
    f"(TPR = {best_tpr:.4f})"
    )

# 4. סיווג מחדש לפי הסף שנמצא
train_pred_b = (df_train['PWM_ratio'] > best_thresh).astype(int)
test_pred_b = (df_test['PWM_ratio'] > best_thresh).astype(int)

# 5. הדפסת confusion matrix לפי הסף החדש
print("\nConfusion Matrix - Training Set (Best Threshold):")
print(confusion_matrix(y_train, train_pred_b))

print("\nConfusion Matrix - Test Set (Best Threshold):")
print(confusion_matrix(y_test, test_pred_b))

# ------------------ סעיף 7 ------------------

# המרת עמודת Group לתוויות בינאריות: Positive = 1, Negative = 0
df['label'] = df['Group'].map({'Positive': 1, 'Negative': 0})


# עיבוד עמודות ratio - נמנע מחלוקה באפס
def safe_ratio(ref, alt):
    return ref / alt if alt != 0 else 0


# יצירת עמודות ratio לכל שיטה
df['PWM_ratio'] = df.apply(
    lambda row: safe_ratio(row['PWM_ref'], row['PWM_alt']), axis=1)
df['MES_ratio'] = df.apply(
    lambda row: safe_ratio(row['MES_ref'], row['MES_alt']), axis=1)
df['NNSPLICE_ratio'] = df.apply(
    lambda row: safe_ratio(row['NNSplice_ref'], row['NNSplice_alt']), axis=1)
df['HSF_ratio'] = df.apply(
    lambda row: safe_ratio(row['HSF_ref'], row['HSF_alt']), axis=1)

# רשימת השיטות להצגה
methods = {
    'PWM': 'PWM_ratio',
    'MES': 'MES_ratio',
    'NNSPLICE': 'NNSPLICE_ratio',
    'HSF': 'HSF_ratio'
}

# ציור גרף ROC לכל שיטה
plt.figure(figsize=(10, 8))
for name, column in methods.items():
    fpr, tpr, _ = roc_curve(df['label'], df[column])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# גרף כולל
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for PWM, MES, NNSPLICE, and HSF')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


# קביעה איזו שיטה הכי טובה לפי AUC
best_method = ''
best_auc = 0
for name, column in methods.items():
    fpr, tpr, _ = roc_curve(df['label'], df[column])
    roc_auc = auc(fpr, tpr)
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_method = name

print(f"\nThe best method based on AUC is: {best_method} "
      f"(AUC = {best_auc:.4f})")
