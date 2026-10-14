with open("code/transfer_learning.py", "r") as f:
    text = f.read()

# Fix scheduler placement
import re
text = text.replace("            pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 'acc': f'{accuracy.item():.4f}'})\n            scheduler.step()",
                    "            pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 'acc': f'{accuracy.item():.4f}'})\n        scheduler.step()")

# Change epochs to 20
text = re.sub(r'epochs=30', 'epochs=20', text)

with open("code/transfer_learning.py", "w") as f:
    f.write(text)
