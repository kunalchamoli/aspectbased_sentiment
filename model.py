import torch as th
import pytorch_lightning as pl
import transformers
import sh
import os
from absl import app, flags, logging
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import trainset, valset, testset

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_string('device', 'cpu', '')
flags.DEFINE_string('modelname', 'bert-base_uncased', '')
flags.DEFINE_integer('batch_size', 16, '')

FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')


class BERT_SPC(pl.LightningModule):
    def __init__(self, dropout = 0.1, bert_dim = 768, polarities_dim = 3):
        super(BERT_SPC, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = th.nn.Dropout(dropout)
        self.dense = th.nn.Linear(bert_dim, polarities_dim)
        self.loss = th.nn.CrossEntropyLoss(reduction= 'none') 

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = [batch[col].to(FLAGS.device) for col in ['text_bert_indices', 'bert_segments_ids']]
        logits = self.forward(inputs)
        target = batch['polarity']
        loss = self.loss(logits, target).mean()

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}       


    def validation_step(self, batch, batch_idx):
        inputs = [batch[col].to(FLAGS.device) for col in ['text_bert_indices', 'bert_segments_ids']]
        logits = self.forward(inputs)
        target = batch['polarity'].to(FLAGS.device)
        loss = self.loss(logits, target)
        acc = (logits.argmax(-1) == batch['polarity']).float()

        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}
    
    def train_dataloader(self):
        return DataLoader(dataset=trainset, batch_size=16, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset = valset, batch_size= 16, shuffle = False )

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(),lr=2e-5, weight_decay=0.01)



def main(_):
    model = BERT_SPC()
    trainer = pl.Trainer(
        default_root_dir= 'logs',
        gpus = (1 if th.cuda.is_available() else 0),
        max_epochs=10,
        fast_dev_run= FLAGS.debug,
        logger= pl.loggers.TensorBoardLogger('logs/', name='aspect', version=0)
    )
    trainer.fit(model)


    if __name__ == "__main__":
        app.run(main)
    