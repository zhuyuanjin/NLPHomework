import argparse
import torch
import data.dict as dict
from data.dataloader import dataset
import data.dataloader as dataloader

parser = argparse.ArgumentParser(description='preprocess.py')
parser.add_argument('-q_length', type = int,  default=30)
parser.add_argument('-a_length', type = int,  default=30)
parser.add_argument('-utter_length', type = int,  default=4)
parser.add_argument('-vocab', default=None)
parser.add_argument('-q_file', default='./data/train/file_q.txt')
parser.add_argument('-r_file', default='./data/train/file_r.txt')
parser.add_argument('-save_data', default='./save_data')

parser.add_argument('-vocab_size', type = int, default=100000)
parser.add_argument('-save_vocab', default=None, help="Path to an existing source vocabulary") 

opt = parser.parse_args()
print(opt)
def makeVocabulary(filename, size, char=False): 
    vocab = dict.Dict([dict.PAD_WORD, dict.UNK_WORD,
                       dict.BOS_WORD, dict.EOS_WORD])
    if char:
        vocab.addSpecial(dict.SPA_WORD)

    # lengths = []

   
    with open(filename,'rb') as f:
        # lines = [line.decode('utf-8').strip().split() for line in txtfile.readlines()]
        for sent in f.readlines():
            # print(sent[0].decode('utf-8'))
            for word in sent.decode('utf-8').strip().split():
                # print(word)
                # lengths.append(len(word))
                # if char:
                #     for ch in word:
                #         vocab.add(ch)
                # else:
                # print(word)
                if word != 'END':
                    vocab.add(word+" ") # why add " " here ?
    # originalSize = vocab.size()
    originalSize = vocab.size()
    vocab = vocab.prune(size) 
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))
    return vocab



def initVocabulary(name, dataFile, vocabFile, vocabSize, char=False):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = dict.Dict()
        vocab.loadFile(vocabFile) 
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize, char=char) 
        vocab = genWordVocab

    # print()
    return vocab

def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)

def makeData(qFile, rFile, Dicts):
    qidx, ridx = [], [] 
    raw_src, raw_tgt = [], [] 
    sizes = [] 
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (qFile, rFile))
    qF = open(qFile,'r',encoding = 'utf-8')
    aF = open(rFile,'r',encoding='utf-8')

    # while True: 
        # qline = qF.readline()
        # aline = aF.readline()
    for aline in aF.readlines():
        # print(len(qline))

        # normal end of file
        # if  len(aline) == 0:
        #     break

        # # source or target does not have same number of lines
        # if len(qline) == 0 or len(aline) == 0:
        #     print('WARNING: source and target do not have the same number of sentences')
        #     break

        # qline = qline.strip()
        aline = aline.strip()
        # print(qline,aline)
        # q and/or r are empty
        # if qline == "" or aline == "":
        #     print('WARNING: ignoring an empty line ('+str(count+1)+')')
        #     ignored += 1
            # continue

        # qWords = qline.decode('utf-8').split()
        aWords = aline.split()
        # print(qWords,aWords)
        # print(qWords)
        # 
        # if opt.q_length == 0 or (len(aWords) <= opt.a_length):        
            # qWords = [word+" " for word in qWords]
        aWords = [word+" " for word in aWords]

        # qidx += [Dicts.convertToIdx(qWords,
                                      # dict.UNK_WORD)] 
        aWords = Dicts.convertToIdx(aWords,dict.UNK_WORD)
        if len(aWords) < opt.a_length:
            aWords.extend([0]*(opt.a_length-len(aWords)))
        else:
            aWords = aWords[-opt.a_length:]
        ridx.append(aWords)
            # raw_src += [srcWords]
            # raw_tgt += [tgtWords]
            # sizes += [len(qWords)] 
        # else:
        #     ignored += 1

        # count += 1
    # print(len(ridx))
    # qF.close()
    qidx = [] 
    qSession=[]
    for line in qF.readlines():
        qline = line.strip()
        # print(qline)
        if qline != 'END':
            qWords = qline.split()
            qWords = [word+" " for word in qWords]
            # print(qWords)
            qWords = Dicts.convertToIdx(qWords, dict.UNK_WORD)
            # print(qWords)
            if len(qWords ) < opt.q_length:
                # [qline.append(0) for i in range(len(q_length) - len(qline))]
                qWords.extend([0]*(opt.q_length-len(qWords)))
            else:
                qWords = qWords[-opt.q_length:]
            # print(len(qWords))
            qSession.append(qWords)

        else:
            # print(len(qSession))
            
            if len(qSession) <= opt.utter_length:
                qSession.extend((opt.utter_length - len(qSession))*[[0]*opt.q_length])
                # print(len(qSession))
            else:
                qSession = qSession[-opt.utter_length:]
            # print(len(qSession))
            qidx.append(qSession)
            qSession = []
            # print(qidx)
            # print(type(qidx))
    # print(qidx)
        # if count % opt.report_every == 0:
        # print('... %d sentences prepared' % count)
    print(len(qidx),len(ridx))
    qF.close()
    aF.close()

    return dataset(torch.Tensor(qidx),torch.Tensor(ridx)) 




def main():
    dicts = {}
    print('share the vocabulary between source and target')
    dicts = initVocabulary('source and target',
                                  opt.q_file,
                                  opt.vocab,
                                  opt.vocab_size)
    # print(dicts.labelToIdx)
    print(dicts.labelToIdx)
    print('Preparing training ...')
    trainset = makeData(opt.q_file, opt.r_file, dicts)
    print(trainset[0][0].size(),trainset[0][1].size())

    trainloader = dataloader.get_loader(trainset, batch_size=4, shuffle=True, num_workers=2)
    for i,a in enumerate(trainloader):
        x,y = a[0],a[1]
        print(x.size(),i)
        # print(y,i,dicts.convertToLabel(y))


    if opt.save_vocab is None:
        saveVocabulary('source', dicts, opt.save_data + '.txt')

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': trainset}
    # print(dicts.labelToIdx)
    torch.save(save_data, opt.save_data) 


if __name__ == "__main__":
    main()