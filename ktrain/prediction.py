from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

def load_dataset(filename):
	data = read_csv(filename,names=["text","class"],header=None)
	return data

# load the dataset
data = load_dataset('train.csv')
data = data.iloc[1:]
data["class"] = labelencoder.fit_transform(data["class"])
total_classes  = labelencoder.classes_

x_train = data["text"].to_list()
y_train = data["class"].to_numpy()

import ktrain
from ktrain import text
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=25, classes=total_classes)
trn = t.preprocess_train(x_train, y_train)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, batch_size=6)

learner.model.load_weights("model.h5")

print("Model Loaded from disk")


predictor = ktrain.get_predictor(learner.model,preproc=t)

prediction = predictor.predict('sample text to test')
print(prediction)
