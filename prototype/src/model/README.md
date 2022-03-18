AudioHUD\prototype\src\model>pyinstaller --onefile --add-data "sample-detections.csv;." model.py

AudioHUD\prototype\src\model>pyinstaller --onefile --add-data "static;." model.py

First run executable.

Then run `node test-client.js'