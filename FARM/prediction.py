
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import MultiLabelTextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=False)

lang_model = "xlnet-large-cased"
do_lower_case = False

tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=lang_model,
    do_lower_case=do_lower_case)


def load_dataset(filename):
    data = read_csv(filename,names=["text","class"],header=None)
    return data

# load the dataset
data = load_dataset('new/train.csv')
data = data.iloc[1:]
data["class"] = labelencoder.fit_transform(data["class"])
label_list  = labelencoder.classes_.tolist()





metric = "f1_macro" # desired metric for evaluation

processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=20, # BERT can only handle sequence lengths of up to 512
                                            label_list=label_list,
                                            data_dir='new/', 
                                            label_column_name="class", # our labels are located in the "genre" column
                                            metric=metric,
                                            quote_char='"',
                                            multilabel=True,
                                            train_filename="train.tsv",
                                            dev_filename=None,
                                            test_filename="test.tsv",
                                            dev_split=0.1 # this will extract 10% of the train set to create a dev set
                                            )


save_dir = "saved_models/my_model_xlm"
#model.save(save_dir)
processor.save(save_dir)

inferenced_model = Inferencer.load(save_dir)

def read_file(file_name: str):
  text_file = open (file_name, 'r')
  text_file = text_file.read().replace('\n', ' ')
  return {'text': text_file}

def create_input(text_files:list):
  model_input = list()
  for text_file in text_files:
    model_input.append(read_file(text_file['file']))
  return model_input

def create_result_overview (articles:list, result:list):
  files = list()
  labels = list()
  predictions = list()
  for i in range(len(articles)):
    predictions.append(result[0]['predictions'][i]['label'].strip("'[]'"))
  
  print(predictions)

article_texts = [{'text': 'random text for testing'}]

result = inferenced_model.inference_from_dicts(article_texts)
print(result[0]['predictions'][0]['label'].strip("'[]'"))

