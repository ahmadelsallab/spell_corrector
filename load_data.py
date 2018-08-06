import os
f_out = open('../dat/all_ocr_data.txt', 'w')
dir = "../dat/DATASETS/"
for dirpath, dirnames, filenames in os.walk(dir):
    for file in filenames:
        if file.endswith(".txt"):
            print(file)
            f = open(os.path.join(dirpath, file))
            '''
            txt = f.read()
            f_out.write(txt)
            '''
            for row in f:
                sents = row.split("\t")
                if len(sents) == 2:
                    f_out.write(row)
            f.close()
