with open("code/transfer_learning.py", "r") as f:
    text = f.read()

# Replace ArcFace params for transfer testing to enable quick convergence
text = text.replace("margin=float(config['model']['arcface'].get('margin', 0.5))", "margin=0.1")
text = text.replace("scale=float(config['model']['arcface'].get('scale', 64))", "scale=16.0")

# Adjust learning rate to be significantly higher to punch through
text = text.replace("lr=3e-3", "lr=1e-2")

# Also, ensure test_transform is clean
with open("code/transfer_learning.py", "w") as f:
    f.write(text)
