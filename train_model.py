from options import Options
from trainer import Trainer
args = Options().parse()

trainer = Trainer(args)
trainer.train()

