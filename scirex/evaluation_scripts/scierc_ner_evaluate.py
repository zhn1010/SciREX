import json
gold = [json.loads(line) for line in open('/home/saeed/Projects/SciREX/scirex_dataset/release_data/test.jsonl')]
predicted = [json.loads(line) for line in open('/home/saeed/Projects/SciREX/test_output/1-ner_predictions.jsonl')]

#--------------------------------------------------------------------------------------------------------#
all_gold_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner']] for doc in gold]
all_gold_num = sum([len(s) for s in all_gold_ner])

method_gold_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner'] if m[2] == 'Method'] for doc in gold]
method_gold_num = sum([len(s) for s in method_gold_ner])

task_gold_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner'] if m[2] == 'Task'] for doc in gold]
task_gold_num = sum([len(s) for s in task_gold_ner])

material_gold_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner'] if m[2] == 'Material'] for doc in gold]
material_gold_num = sum([len(s) for s in material_gold_ner])

metric_gold_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner'] if m[2] == 'Metric'] for doc in gold]
metric_gold_num = sum([len(s) for s in metric_gold_ner])
#--------------------------------------------------------------------------------------------------------#
all_predicted_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner']] for doc in predicted]
all_predicted_num = sum([len(s) for s in all_predicted_ner])

method_predicted_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner'] if m[2] == 'Method'] for doc in predicted]
method_predicted_num = sum([len(s) for s in method_predicted_ner])

task_predicted_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner'] if m[2] == 'Task'] for doc in predicted]
task_predicted_num = sum([len(s) for s in task_predicted_ner])

material_predicted_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner'] if m[2] == 'Material'] for doc in predicted]
material_predicted_num = sum([len(s) for s in material_predicted_ner])

metric_predicted_ner = [[(m[0], m[1] - 1, m[2]) for m in doc['ner'] if m[2] == 'Metric'] for doc in predicted]
metric_predicted_num = sum([len(s) for s in metric_predicted_ner])
#--------------------------------------------------------------------------------------------------------#

all_matched = sum([len(set(g) & set(p)) for g, p in zip(all_predicted_ner, all_gold_ner)])
method_matched = sum([len(set(g) & set(p)) for g, p in zip(method_predicted_ner, method_gold_ner)])
material_matched = sum([len(set(g) & set(p)) for g, p in zip(material_predicted_ner, material_gold_ner)])
metric_matched = sum([len(set(g) & set(p)) for g, p in zip(metric_predicted_ner, metric_gold_ner)])
task_matched = sum([len(set(g) & set(p)) for g, p in zip(task_predicted_ner, task_gold_ner)])

p = all_matched / all_predicted_num
r = all_matched / all_gold_num
f1 = 2 * p * r / (p + r)

print("ALL")
print(f"p = {p}")
print(f"r = {r}")
print(f"f1 = {f1}")

p = method_matched / method_predicted_num
r = method_matched / method_gold_num
f1 = 2 * p * r / (p + r)

print("METHOD")
print(f"p = {p}")
print(f"r = {r}")
print(f"f1 = {f1}")

p = material_matched / material_predicted_num
r = material_matched / material_gold_num
f1 = 2 * p * r / (p + r)

print("MATERIAL")
print(f"p = {p}")
print(f"r = {r}")
print(f"f1 = {f1}")

p = metric_matched / metric_predicted_num
r = metric_matched / metric_gold_num
f1 = 2 * p * r / (p + r)

print("DATASET")
print(f"p = {p}")
print(f"r = {r}")
print(f"f1 = {f1}")

p = task_matched / task_predicted_num
r = task_matched / task_gold_num
f1 = 2 * p * r / (p + r)

print("TASK")
print(f"p = {p}")
print(f"r = {r}")
print(f"f1 = {f1}")