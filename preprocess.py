import argparse
import torch
import data.dict as dict
from data.dataloader import dataset
import data.dataloader
parser = argparse.ArgumentParser(description='preprocess.py')
parser.add_argument('-q_length', type = int,  default=100)
parser.add_argument('-a_length', type = int,  default=100)
parser.add_argument('-vocab', default=None)
parser.add_argument('-q_file', default='./data/file_q.txt')
parser.add_argument('-r_file', default='./data/file_r.txt')
parser.add_argument('-save_data', default='./save_data')

parser.add_argument('-vocab_size', type = int, default=50000)
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
                vocab.add(word+" ") # why add " " here ?
    # originalSize = vocab.size()
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

    print()
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
    qF = open(qFile,'rb')
    aF = open(rFile,'rb')

    while True: 
        qline = qF.readline()
        aline = aF.readline()
        # print(len(qline))

        # normal end of file
        if len(qline) == 0 and len(aline) == 0:
            break

        # source or target does not have same number of lines
        if len(qline) == 0 or len(aline) == 0:
            print('WARNING: source and target do not have the same number of sentences')
            break

        qline = qline.strip()
        aline = aline.strip()
        # print(qline,aline)
        # q and/or r are empty
        if qline == "" or aline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            ignored += 1
            continue

        qWords = qline.decode('utf-8').split()
        aWords = aline.decode('utf-8').split()
        # print(qWords,aWords)
        # print(qWords)
        # 
        if opt.q_length == 0 or (len(qWords) <= opt.q_length and len(aWords) <= opt.a_length):        
            qWords = [word+" " for word in qWords]
            aWords = [word+" " for word in aWords]

            qidx += [Dicts.convertToIdx(qWords,
                                          dict.UNK_WORD)] 
            ridx += [Dicts.convertToIdx(aWords,
                                          dict.UNK_WORD)]
            # raw_src += [srcWords]
            # raw_tgt += [tgtWords]
            sizes += [len(qWords)] 
        else:
            ignored += 1

        count += 1

        # if count % opt.report_every == 0:
        # print('... %d sentences prepared' % count)

    qF.close()
    aF.close()

    return dataset(qidx,ridx) 




def main():
    dicts = {}
    print('share the vocabulary between source and target')
    dicts = initVocabulary('source and target',
                                  opt.q_file,
                                  opt.vocab,
                                  opt.vocab_size)
    # print(dicts.labelToIdx)

    print('Preparing training ...')
    trainset = makeData(opt.q_file, opt.r_file, dicts)
    # print(trainset[1])

    # trainloader = dataloader.get_loader(trainset, batch_size=3, shuffle=True, num_workers=2)
    # for i,a in enumerate(trainloader):
    #     x,y = a[0],a[1]
    #     # print(x,i)
    #     # print(y,i,dicts.convertToLabel(y))


    if opt.save_vocab is None:
        saveVocabulary('source', dicts, opt.save_data + '.txt')

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': trainset}
    torch.save(save_data, opt.save_data) 


if __name__ == "__main__":
    main()