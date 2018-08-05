import numpy as np
import csv

def calculate_WER_sent(gt, pred):
    '''
    calculate_WER('calculating wer between two sentences', 'calculate wer between two sentences')
    '''
    gt_words = gt.lower().split(' ')
    pred_words = pred.lower().split(' ')
    d = np.zeros(((len(gt_words)+1),(len(pred_words)+1)), dtype=np.uint8)
    #d = d.reshape((len(gt_words)+1, len(pred_words)+1))
    
    #Initializing error matrix
    for i in range(len(gt_words)+1):
        for j in range(len(pred_words)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(gt_words)+1):
        for j in range(1, len(pred_words)+1):
            if gt_words[i-1] == pred_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(gt_words)][len(pred_words)]


def calculate_WER(gt, pred):
    '''

    :param gt: list of sentences of the ground truth
    :param pred: list of sentences of the predictions
    both lists must have the same length
    :return: accumulated WER
    '''
    assert len(gt)==len(pred)
    WER = 0
    nb_w = 0
    for i in range(len(gt)):
        WER += calculate_WER_sent(gt[i], pred[i])
        nb_w += len(gt[i])

    return  WER/nb_w

tesseract_output = []
spell_corrected = []
gt = []
for row in open('..\data_for_WER.txt'):
    sents = row.split("\t")
    tesseract_output.append(sents[0])
    spell_corrected.append(sents[1])
    gt.append(sents[2])


WER_tesseract = calculate_WER(gt, tesseract_output)
print('WER_tesseract = ', WER_tesseract)

WER_spell_correction = calculate_WER(gt, spell_corrected)
print('WER_spell_correction = ', WER_spell_correction)

# Now use another spell checker
from autocorrect import spell
for i, sent in  enumerate(spell_corrected):
    corr_sent = []
    for word in sent.split(' '):
        corr_sent.append(spell(word))

    spell_corrected[i] = ' '.join(corr_sent)

WER_tesseract = calculate_WER(gt, tesseract_output)
print('WER_tesseract = ', WER_tesseract)

WER_spell_correction = calculate_WER(gt, spell_corrected)
print('WER_spell_correction = ', WER_spell_correction)