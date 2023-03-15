


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
        
    
    def forward (self, input_graph, **kwargs):
        x = self.model(x, edge_index)
        
        self.loss_function
        
        
    def training_step (self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss
    
    def configure_optimizers (self):
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, 
                              weight_decay=2e-3)
        return optimizer
    
    def augmentation (self):
        pass