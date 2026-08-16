"""
FP-Growth Algorithm - Supermarket Transactions
Data Mining Project

This script finds frequent itemsets and association rules
from supermarket purchase data using the FP-Growth algorithm.
"""

# ─────────────────────────────────────────────
# 1. Import Libraries
# ─────────────────────────────────────────────
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# ─────────────────────────────────────────────
# 2. Load the Dataset
# ─────────────────────────────────────────────
print("=" * 55)
print("  FP-Growth - Supermarket Transactions")
print("=" * 55)

data = pd.read_csv("supermarket_transactions.csv")

# Drop the TransactionID column — it's just an identifier, not a product
data = data.drop(columns=["TransactionID"])

print(f"\n Dataset loaded successfully!")
print(f"   Total transactions : {data.shape[0]}")
print(f"   Total products     : {data.shape[1]}")
print(f"\n First 5 rows:")
print(data.head())

# ─────────────────────────────────────────────
# 3. Data Overview — How often is each item bought?
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print(" Item Purchase Frequency")
print("─" * 55)

item_counts = data.sum().sort_values(ascending=False)
print(f"\n{'Item':<12} {'Count':>6} {'Support %':>10}")
print("-" * 30)
for item, count in item_counts.items():
    support_pct = (count / len(data)) * 100
    print(f"{item:<12} {int(count):>6} {support_pct:>9.1f}%")

# ─────────────────────────────────────────────
# 4. Make Sure the Data is Boolean (True/False)
# ─────────────────────────────────────────────
# FP-Growth requires boolean values (True = bought, False = not bought)
data = data.astype(bool)

# ─────────────────────────────────────────────
# 5. Apply FP-Growth to Find Frequent Itemsets
# ─────────────────────────────────────────────
# min_support = 0.4 means: only keep itemsets bought in at least 40% of transactions
print("\n" + "─" * 55)
print(" Step 1: Finding Frequent Itemsets (min support = 40%)")
print("─" * 55)

frequent_itemsets = fpgrowth(data, min_support=0.4, use_colnames=True)

# Sort by support (highest first) so we see the most common patterns
frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

print(f"\n Found {len(frequent_itemsets)} frequent itemsets!\n")
print(f"{'Itemset':<35} {'Support':>8}")
print("-" * 45)
for _, row in frequent_itemsets.iterrows():
    items = ", ".join(list(row["itemsets"]))
    print(f"{items:<35} {row['support']:>7.1%}")

# ─────────────────────────────────────────────
# 6. Generate Association Rules
# ─────────────────────────────────────────────
# min_threshold = 0.6 means: only keep rules with confidence >= 60%
# Confidence = how often the rule is correct when the antecedent is present
print("\n" + "─" * 55)
print(" Step 2: Generating Association Rules (min confidence = 60%)")
print("─" * 55)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Sort by lift (best rules first)
# Lift > 1 means the items appear together MORE than by chance
rules = rules.sort_values("lift", ascending=False)

print(f"\n Found {len(rules)} association rules!\n")
print(f"{'IF (antecedent)':<18} {'THEN (consequent)':<18} {'Confidence':>11} {'Lift':>7}")
print("-" * 58)
for _, row in rules.iterrows():
    antecedent = ", ".join(list(row["antecedents"]))
    consequent = ", ".join(list(row["consequents"]))
    print(f"{antecedent:<18} {consequent:<18} {row['confidence']:>10.1%} {row['lift']:>7.2f}")

# ─────────────────────────────────────────────
# 7. Highlight the Top 5 Rules (Easiest to Explain)
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print(" Top 5 Strongest Rules by Lift")
print("─" * 55)

top5 = rules.head(5)
print()
for i, (_, row) in enumerate(top5.iterrows(), 1):
    antecedent = ", ".join(list(row["antecedents"]))
    consequent = ", ".join(list(row["consequents"]))
    print(f"  Rule #{i}:")
    print(f"    IF customer buys  → {antecedent}")
    print(f"    THEN they also buy → {consequent}")
    print(f"    Confidence: {row['confidence']:.1%}  |  Lift: {row['lift']:.2f}")
    print()

# ─────────────────────────────────────────────
# 8. Save Results to CSV
# ─────────────────────────────────────────────
frequent_itemsets_out = frequent_itemsets.copy()
frequent_itemsets_out["itemsets"] = frequent_itemsets_out["itemsets"].apply(
    lambda x: ", ".join(list(x))
)
frequent_itemsets_out.to_csv("frequent_itemsets.csv", index=False)

rules_out = rules.copy()
rules_out["antecedents"] = rules_out["antecedents"].apply(lambda x: ", ".join(list(x)))
rules_out["consequents"] = rules_out["consequents"].apply(lambda x: ", ".join(list(x)))
rules_out[["antecedents", "consequents", "support", "confidence", "lift"]].to_csv(
    "association_rules.csv", index=False
)

print("─" * 55)
print(" Results saved!")
print("   frequent_itemsets.csv")
print("   association_rules.csv")
print("─" * 55)
