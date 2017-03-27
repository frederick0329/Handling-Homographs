import argparse
from random import shuffle
import os
from time import gmtime, strftime

# output number of times model2 win over model1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--src', help='src language file', required=True)
    parser.add_argument('-tgt', '--tgt', help='target language file', required=True)
    parser.add_argument('-model1', '--model1', help='model1 path', required=True)
    parser.add_argument('-model2', '--model2', help='model2 path', required=True)
    #parser.add_argument('-concat', '--concat', default=False, help='concat or not')
    #parser.add_argument('-context_type', '--context_type', default='contextBiEncoder', help='context type')
    parser.add_argument('-gpuid', '--gpuid', default='1', help='gpu id')

    parser.add_argument('-M', '--M', default=1000, help='number of trials')
    parser.add_argument('-ratio', '--ratio', default=0.5, help='ratio of original instances')

    parser.add_argument('-out', '--out', help='significant test output path', required=True)
    opts = parser.parse_args()

    command_translate1 = 'th translate.lua -model ' + opts.model1 + \
                                                          ' -src ' + opts.src + \
                                                          ' -output sigTestTmpFiles/sigTest_pred1.txt' + \
                                                          ' -gpuid ' + opts.gpuid + ' -batch_size 400 -replace_unk -disable_logs'
    command_translate2 = 'th translate.lua -model ' + opts.model2 + \
                                                          ' -src ' + opts.src + \
                                                          ' -output sigTestTmpFiles/sigTest_pred2.txt' + \
                                                          ' -gpuid ' + opts.gpuid + ' -batch_size 400 -replace_unk -disable_logs' + \
                                                          ' -concat -gating_type contextBiEncoder'

    command_eval1 = 'perl ./benchmark/3rdParty/multi-bleu.perl sigTestTmpFiles/sigTest_tmp_tgt.txt < sigTestTmpFiles/sigTest_tmp_pred1.txt > sigTestTmpFiles/sigTest_tmp_eval1.txt'
    command_eval2 = 'perl ./benchmark/3rdParty/multi-bleu.perl sigTestTmpFiles/sigTest_tmp_tgt.txt < sigTestTmpFiles/sigTest_tmp_pred2.txt > sigTestTmpFiles/sigTest_tmp_eval2.txt'
    print 'model 1 translating...'
    #os.system(command_translate1)
    print 'model 2 translating...'
    #os.system(command_translate2)

    data_tgt = open(opts.tgt).readlines()
    pred1 = open('sigTestTmpFiles/sigTest_pred1.txt').readlines()
    pred2 = open('sigTestTmpFiles/sigTest_pred2.txt').readlines()

    win = 0.0
    lose = 0.0
    tie = 0.0
    for t in xrange(int(opts.M)):
        combined = list(zip(data_tgt, pred1, pred2))
        shuffle(combined)
        data_tgt[:], pred1[:], pred2[:] = zip(*combined)


        writer_tgt = open('sigTestTmpFiles/sigTest_tmp_tgt.txt', 'w')
        writer_pred1 = open('sigTestTmpFiles/sigTest_tmp_pred1.txt', 'w')
        writer_pred2 = open('sigTestTmpFiles/sigTest_tmp_pred2.txt', 'w')
        for i in xrange(int(opts.ratio*len(data_tgt))):
            writer_tgt.write(data_tgt[i])
            writer_pred1.write(pred1[i])
            writer_pred2.write(pred2[i])
        writer_tgt.close()
        writer_pred1.close()
        writer_pred2.close()

        os.system(command_eval1)
        os.system(command_eval2)

        score1 = float(open('sigTestTmpFiles/sigTest_tmp_eval1.txt', 'r').readlines()[0].split(',')[0].split('=')[1])
        score2 = float(open('sigTestTmpFiles/sigTest_tmp_eval2.txt', 'r').readlines()[0].split(',')[0].split('=')[1])
        if score2 > score1:
            win += 1.0
        elif score2 == score1:
            tie += 1.0
        else:
            lose += 1.0

        print strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '   ' + 'win: ' + str(win) + ' lose: ' + str(lose) + ' tie: ' + str(tie) + ' score1: ' + str(score1) + ' score2: ' + str(score2)

    
    open(opts.out, 'w').write('win: ' + str(win/float(t+1)) + ' lose: ' + str(lose/float(t+1)) + ' tie: ' + str(tie/float(t+1)))
