from PIL import Image
import os

for folderPath, dirs, files  in os.walk('data'):
    if dirs:
        continue
    newFolder = os.path.join(folderPath, 'compressed')
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)
    else:
        continue
    for file in files:
        filePath = os.path.join(folderPath, file)
        img = Image.open(filePath)
        img = img.resize((256, 256))
        img.save(os.path.join(newFolder, file))
