path = r"C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai\src\data_loading\lidc_loader.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
old = "pad: tuple = ((20, 20), (20, 20), (0, 0)),"
new = "pad: list = [(20, 20), (20, 20), (0, 0)],"
assert old in content, "pattern not found"
content = content.replace(old, new)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("patched")
