with open("code/transfer_learning.py", "r") as f:
    text = f.read()

text = text.replace("    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)", "    epochs=30\n    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)")
text = text.replace("    epochs = 30\n    model.train()", "    model.train()")

with open("code/transfer_learning.py", "w") as f:
    f.write(text)
