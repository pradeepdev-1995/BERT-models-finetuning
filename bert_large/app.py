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

# Create a ClassificationModel
model = ClassificationModel(
    'bert',
    'bert-large-uncased-whole-word-masking-finetuned-squad',
    num_labels=total_classes,
    use_cuda=False,
    args={'learning_rate':1e-5,'save_model_every_epoch': False,'num_train_epochs': 111, 'reprocess_input_data': True, 'overwrite_output_dir': False}
) 

# Train the model
model.train_model(data,output_dir=None)