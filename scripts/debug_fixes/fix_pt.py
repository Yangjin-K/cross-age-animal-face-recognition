import re

with open('code/progressive_transfer.py', 'r') as f:
    content = f.read()

# Add logging
content = "import logging\nfrom datetime import datetime\n" + content

main_block = """def main():
    parser = argparse.ArgumentParser(description='Transfer Learning for Human Face Datasets')"""

new_main_block = """def main():
    parser = argparse.ArgumentParser(description='Transfer Learning for Human Face Datasets')"""

content = content.replace(main_block, new_main_block)

# Let's just fix it by replacing the main function and adding log setup:
setup_code = """
    # Setup logger
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"progressive_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    
    # Overwrite print with logging
    global print
    def print(*args, **kwargs):
        msg = " ".join(map(str, args))
        logging.info(msg)
"""

content = content.replace("    args = parser.parse_args()", "    args = parser.parse_args()\n" + setup_code)

# Increase scale, tweak margin
content = content.replace("margin=0.5,\n        scale=32.0,", "margin=0.45,\n        scale=64.0,")

with open('code/progressive_transfer.py', 'w') as f:
    f.write(content)
