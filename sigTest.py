import argparse
from random import shuffle
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--src', help='src language file', required=True)
    parser.add_argument('-tgt', '--tgt', help='target language file', required=True)
    parser.add_argument('-model', '--model', help='model path', required=True)
    parser.add_argument('-concat', '--concat', default=False, help='concat or not')
    parser.add_argument('-context_type', '--context_type', default='contextBiEncoder', help='context type')
    parser.add_argument('-gpuid', '--gpuid', default='1', help='gpu id')

    parser.add_argument('-M', '--M', default=1000, help='number of trials')
    parser.add_argument('-ratio', '--ratio', default=0.5, help='ratio of original instances')

    parser.add_argument('-out', '--out', help='significant test output path', required=True)
    opts = parser.parse_args()

    command_translate = 'th translate.lua -model ' + opts.model + \
                                        ' -src sigTest_tmp_input.txt' + \
                                        ' -output sigTest_tmp_pred.txt' + \
                                        ' -gpuid ' + opts.gpuid + ' -batch_size 400 -replace_unk -disable_logs'
    if opts.concat:
        command_translate += ' -concat -gating_type ' + opts.context_type

    command_eval = 'perl ./benchmark/3rdParty/multi-bleu.perl ' + opts.tgt + ' < sigTest_tmp_pred.txt > sigTest_tmp_eval.txt'


    data = open(opts.src).readlines()

    scores = []
    for t in xrange(int(opts.M)):
        print t, opts.M
        shuffle(data)
        writer = open('sigTest_tmp_input.txt', 'w')
        for i in xrange(int(opts.ratio*len(data))):
            writer.write(data[i])
        writer.close()
        os.system(command_translate)
        os.system(command_eval)
        score = float(open('sigTest_tmp_eval.txt', 'r').readlines()[0].split(',')[0].split('=')[1])
        scores.append(score)

    scores = sorted(scores)
    lower = scores[int(float(opts.M)*0.025)]
    upper = scores[int(float(opts.M)*0.975)]
    
    open(opts.out, 'w').write('['+str(lower)+','+str(upper)+']')
