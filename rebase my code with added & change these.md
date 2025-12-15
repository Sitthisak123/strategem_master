*rebase my code with added & change these features:
-change src-img from screeenshot to load image file from "./src/img/*" sorted by name
-load data from "./strategems.csv" 
-add save strategems-icon to "./output" folder matched name and named by Index column in "./strategems.csv" replace output if exist
-add log file "./output/log.txt" to save skipped files (not found in strategems.csv) and log progress in realtime


#main.py, test.py, src\strategems.csv, 
*merge my code with these:
-in main.py change ocr to matchTemplate (ref in ./test.py) with same ability 
-template img from screenshot
-src img from "./img/*"
-indexing is in src\strategems.csv