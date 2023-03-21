


import pytorch_lightning as pl
from torch import optim,Tensor


class EmbeddingTrainer(pl.LightningModule):

    def __init__(self,
                 model,
                 loss_function,
                 params: dict) -> None:
        super(EmbeddingTrainer, self).__init__()

        self.model = model
        self.params = params
        self.loss_function = loss_function
        
    
    def forward (self, input_graph, mode="test"):
        return self.model(input_graph, mode)
        
    def training_step (self, batch, batch_idx):
        mp_result, sc_result = self.forward(batch, mode="train")
        loss = self.loss_function(mp_result, sc_result)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers (self):
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, 
                              weight_decay=2e-3)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        mp_result, sc_result = self.forward(batch, mode="train")
        loss = self.loss_function(mp_result, sc_result)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        mp_result = self.forward(batch, mode="test")