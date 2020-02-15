import dill as pkl
'''
with open('caches/kern_vcr.pkl', 'rb') as f:
    all_pred_entries = pkl.load(f)
'''
print("hi")
with open('caches/kern_sgcls.pkl', 'rb') as f:
    all_pred_entries = pkl.load(f)
print("bye")
print(all_pred_entries[0])
