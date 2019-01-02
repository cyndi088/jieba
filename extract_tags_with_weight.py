import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags_with_weight.py [file name] -k [top k] -w [with weight=1 or 0]"

parser = OptionParser(USAGE)
parser.add_option("-k", dest="topK")
parser.add_option("-w", dest="withWeight")
opt, args = parser.parse_args()


if len(args) < 1:
    print(USAGE)
    sys.exit(1)

file_name = args[0]

if opt.topK is None:
    topK = 10
else:
    topK = int(opt.topK)

if opt.withWeight is None:
    withWeight = False
else:
    if int(opt.withWeight) is 1:
        withWeight = True
    else:
        withWeight = False

# content = open(file_name, 'rb').read()
content = '习近平对食品安全工作作出重要指示中共中央总书记、国家主席、中央军委主席习近平，日前对食品安全工作作出重要指示指出'


# tags = jieba.analyse.extract_tags(content, topK=topK, withWeight=withWeight)
tags = jieba.analyse.textrank(content, topK=topK, withWeight=withWeight)


if withWeight is True:
    for tag in tags:
        print("tag: %s\t\t weight: %f" % (tag[0], tag[1]))
else:
    print(",".join(tags))
