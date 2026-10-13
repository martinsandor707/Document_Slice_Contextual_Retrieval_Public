from statsmodels.stats.contingency_tables import mcnemar

# Contingency Table Structure:
# [[Both Correct,   User Correct/Base Wrong],
#  [User Wrong/Base Correct, Both Wrong]]

# Worst-case assumption: Your model is a 'subset' of the baseline
# b = 0 (You never beat the baseline when it fails)
# c = 2 (The baseline beat you 1 time)
table = [[236, 0], 
         [1, 13]]

result = mcnemar(table, exact=True)

print("McNemar's Test Result for highly overlapping errors:")
print(f"Statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")
# Output: P-value: 0.5

table = [[223, 13], 
         [14, 0]]

result = mcnemar(table, exact=True)

print("McNemar's Test Result for NO overlapping errors:")
print(f"Statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")