import urllib.request
import shutil
import os

# create standard file locations (corresponding to default settings.py settings)
if not os.path.exists("nn_input_data"):
    os.makedirs("nn_input_data")
if not os.path.exists("nn_input_data/wikiconv"):
    os.makedirs("nn_input_data/wikiconv")
if not os.path.exists("nn_input_data/cmv"):
    os.makedirs("nn_input_data/cmv")

url = "http://zissou.infosci.cornell.edu/convokit/models/craft_wikiconv/train_processed_dialogs.txt"
with urllib.request.urlopen(url) as response, open("nn_input_data/wikiconv/train_processed_dialogs.txt", "wb") as out_file:
    length = float(response.info()["Content-Length"])
    length = str(round(length / 1e6, 1)) + "MB" \
        if length > 1e6 else \
        str(round(length / 1e3, 1)) + "KB"
    print("Downloading wikiconv training data from", url, "(" + length + ")...", end=" ", flush=True)
    shutil.copyfileobj(response, out_file)

url = "http://zissou.infosci.cornell.edu/convokit/models/craft_cmv/train_processed_dialogs.txt"
with urllib.request.urlopen(url) as response, open("nn_input_data/cmv/train_processed_dialogs.txt", "wb") as out_file:
    length = float(response.info()["Content-Length"])
    length = str(round(length / 1e6, 1)) + "MB" \
        if length > 1e6 else \
        str(round(length / 1e3, 1)) + "KB"
    print("Downloading CMV training data from", url, "(" + length + ")...", end=" ", flush=True)
    shutil.copyfileobj(response, out_file)