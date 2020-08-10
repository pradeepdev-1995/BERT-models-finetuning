
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
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

def load_dataset(filename):
    data = read_csv(filename,names=["text","class"],header=None)
    return data

# load the dataset
data = load_dataset('new/train.csv')
data = data.iloc[1:]
data["class"] = labelencoder.fit_transform(data["class"])
label_list  = labelencoder.classes_.tolist()

data = read_csv("new/train.csv",names=["text","class"],header=None)
data = data.iloc[1:]

outfile = "new/train.tsv"
data.to_csv(outfile,sep='\t',index=False)

outfile = "new/test.tsv"
data.to_csv(outfile,sep='\t',index=False)


set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=False)
n_epochs = 111
batch_size = 35 # larger batch sizes might use too much computing power in Colab


lang_model = "xlnet-large-cased"
do_lower_case = False

tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=lang_model,
    do_lower_case=do_lower_case)

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

data_silo = DataSilo(
    processor=processor,
    batch_size=batch_size)

# loading the pretrained BERT base cased model
language_model = LanguageModel.load(lang_model)
# prediction head for our model that is suited for classifying news article genres
prediction_head = MultiLabelTextClassificationHead(num_labels=len(label_list))

model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs)

trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        device=device)

trainer.train()

save_dir = "saved_models/my_model_xlm"
model.save(save_dir)
processor.save(save_dir)

inferenced_model = Inferencer.load(save_dir)