"""
Define common constants.
"""
TRAIN_JSON = 'train.json'
DEV_JSON = 'dev.json'
TEST_JSON = 'test.json'

GLOVE_DIR = 'dataset/glove'

EMB_INIT_RANGE = 1.0
MAX_LEN = 100

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

TITLE_LABEL_TO_ID = {'no_relation': 0, 'per:title': 1}
TOP_MEMBERS_LABEL_TO_ID = {'no_relation': 0, 'org:top_members/employees': 1}
EMPLOYEE_LABEL_TO_ID = {'no_relation': 0, 'per:employee_of': 1}


TITLE_RULES = {
    'OBJ-TITLE SUBJ-PERSON': [1 - 0.916435, 0.916435],
    'OBJ_RIGHT_PERSON': [1 - 0.0826446, 0.0826446],
    'SUBJ-PERSON , OBJ-TITLE': [1 - 0.833333, 0.833333],
    'SUBJ-PERSON , DT OBJ-TITLE': [1 - 0.933333, 0.933333],
    'OBJ_LEFT_ORGANIZATION': [1 - 0.701493, 0.701493],
    'SUBJ-PERSON , DT JJ OBJ-TITLE': [1 - 0.939394, 0.939394],
    'OBJ_LEFT_MISC': [1 - 0.686747, 0.686747],
    'SUBJ-PERSON , the OBJ-TITLE': [1 - 0.857143, 0.857143],
    'SUBJ-PERSON , DT NN OBJ-TITLE': [1 - 0.666667, 0.666667],
    'OBJ-TITLE , SUBJ-PERSON': [1 - 0.76, 0.76],
    'OBJ_LEFT_LOCATION': [1 - 0.783784, 0.783784],
    'SUBJ-PERSON , a OBJ-TITLE': [1 - 1.0, 1.0]
}

TOP_MEMBERS_RULES = {
    'SUBJ-ORGANIZATION NN OBJ-PERSON': [1 - 0.484848, 0.484848],
    'OBJ-PERSON , JJ NN IN DT SUBJ-ORGANIZATION': [1 - 0.9, 0.9],
    'OBJ-PERSON , DT SUBJ-ORGANIZATION': [1 - 0.7, 0.7],
    'OBJ-PERSON , executive director of the SUBJ-ORGANIZATION': [1 - 0.8, 0.8],
    'OBJ-PERSON , the SUBJ-ORGANIZATION': [1 - 0.666667, 0.666667],
    'OBJ-PERSON , DT NN IN DT SUBJ-ORGANIZATION': [1 - 0.479167, 0.479167],
    'OBJ-PERSON VBD DT SUBJ-ORGANIZATION': [1 - 0.0, 0.0],
    'OBJ-PERSON of SUBJ-ORGANIZATION': [1 - 0.5, 0.5],
    'OBJ-PERSON IN SUBJ-ORGANIZATION': [1 - 0.4, 0.4],
    'OBJ-PERSON , NN IN DT SUBJ-ORGANIZATION': [1 - 0.881579, 0.881579],
    'SUBJ-ORGANIZATION NNP OBJ-PERSON': [1 - 1.0, 1.0],
    'SUBJ-ORGANIZATION spokesman OBJ-PERSON': [1 - 0.666667, 0.666667],
    'OBJ-PERSON , SUBJ-ORGANIZATION': [1 - 0.470588, 0.470588],
    'OBJ-PERSON VBD SUBJ-ORGANIZATION': [1 - 0.0, 0.0],
    'SUBJ-ORGANIZATION , OBJ-PERSON': [1 - 0.555556, 0.555556],
    'OBJ-PERSON , president of the SUBJ-ORGANIZATION': [1 - 0.944444, 0.944444]
}

EMPLOYEE_RULES = {
    'SUBJ-PERSON , NN IN DT OBJ-ORGANIZATION': [1 - 1.0, 1.0],
    'OBJ-ORGANIZATION NN SUBJ-PERSON': [1 - 1.0, 1.0],
    'OBJ-ORGANIZATION spokesman SUBJ-PERSON': [1 - 1.0, 1.0],
    'OBJ-ORGANIZATION , SUBJ-PERSON': [1 - 1.0, 1.0],
    'OBJ_RIGHT_DATE': [1 - 0.428571, 0.428571],
    'SUBJ-PERSON , OBJ-ORGANIZATION': [1 - 1.0, 1.0],
    'SUBJ_LEFT_DATE': [1 - 0.0, 0.0],
    'SUBJ_LEFT_PERSON': [1 - 1.0, 1.0],
    'SUBJ-PERSON VBD DT OBJ-ORGANIZATION': [1 - 1.0, 1.0],
    'SUBJ-PERSON , DT NN IN OBJ-ORGANIZATION': [1 - 1.0, 1.0],
    'SUBJ-PERSON , OBJ-LOCATION': [1 - 0.0, 0.0],
    'OBJ-ORGANIZATION SUBJ-PERSON': [1 - 1.0, 1.0],
    'OBJ-ORGANIZATION NNP SUBJ-PERSON': [1 - 1.0, 1.0],
    'OBJ-ORGANIZATION chief SUBJ-PERSON': [1 - 1.0, 1.0],
    'OBJ-ORGANIZATION NNP NNP SUBJ-PERSON': [1 - 1.0, 1.0],
}