with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# cleanup
text = text.replace("\r", "")