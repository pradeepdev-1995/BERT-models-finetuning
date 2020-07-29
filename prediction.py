from simpletransformers.classification import ClassificationModel
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

def load_dataset(filename):
	data = read_csv(filename,names=["questions","tile"],header=None)
	return data

# load the dataset
data = load_dataset('train.csv')
data = data.iloc[1:]
data["tile"] = labelencoder.fit_transform(data["tile"])
total_classes  = len(labelencoder.classes_)

bert_base_model = ClassificationModel('bert', '/bert_base/outputs', num_labels=total_classes,
    use_cuda=False)

albert_model = ClassificationModel('albert', 'albert/outputs', num_labels=total_classes,
    use_cuda=False)

distilbert_model = ClassificationModel('distilbert', '/distilbert/outputs', num_labels=total_classes,
    use_cuda=False)

roberta_model = ClassificationModel('roberta', '/roberta/outputs', num_labels=total_classes,
    use_cuda=False)

bert_large_model = ClassificationModel('bert', '/bert_large/outputs', num_labels=total_classes,
    use_cuda=False)

xlnet_model = ClassificationModel('xlnet', '/XLNet/outputs', num_labels=total_classes,
    use_cuda=False)


query = ["Name the scar-faced bounty hunter of The Old West"]

print("bert_base prediction")
prediction, raw_outputs = bert_base_model.predict(query)
predicted_tile = labelencoder.inverse_transform(prediction)
print(predicted_tile)

print("albert prediction")
prediction, raw_outputs = albert_model.predict(query)
predicted_tile = labelencoder.inverse_transform(prediction)
print(predicted_tile)

print("distilbert prediction")
prediction, raw_outputs = distilbert_model.predict(query)
predicted_tile = labelencoder.inverse_transform(prediction)
print(predicted_tile)

print("roberta prediction")
prediction, raw_outputs = roberta_model.predict(query)
predicted_tile = labelencoder.inverse_transform(prediction)
print(predicted_tile)

print("bert_large prediction")
prediction, raw_outputs = bert_large_model.predict(query)
predicted_tile = labelencoder.inverse_transform(prediction)
print(predicted_tile)

print("xlnet prediction")
prediction, raw_outputs = xlnet_model.predict(query)
predicted_tile = labelencoder.inverse_transform(prediction)
print(predicted_tile)