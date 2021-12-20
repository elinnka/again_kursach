import nltk
nltk.download('stopwords')
from models import *
from demo import *


#Считывание векторов характеристик из файла
data = pd.read_csv("/content/drive/MyDrive/table_final3_754115_5825.csv")
data = data.drop(columns="Unnamed: 0")
data = data.drop_duplicates().reset_index().drop(columns="index")

# исправление ошибок прошлого
for i in range(len(data)):
    if data['root_match'][i] == 'False':
        data.loc[i, 'root_match'] = 0
filename = "attempthelp_5825_1000.csv"
data_text = pd.read_csv(filename)
answer_prepared = data_text['answer_prep']
filename = "attempthelp_5825_1000.csv"
data_text = pd.read_csv(filename)
indexOfContext = []
for i in range(len(data_text)):
    indexOfContext.append(list(data_text["context"].drop_duplicates().reset_index(drop=True)).index(data_text['context'][i]))
# закончено исправление


# Случайное угадывание как бейзлайн
print("Результат на случайном угадывании: ", baseline.F1_accuracy(data))
lg = Logreg(data, answer_prepared, indexOfContext)
print("", lg.evaluate())

# демонстрация
demo_context_beg = '''The choice of Princess Sophie as wife of the future tsar was one result of the Lopukhina Conspiracy in which Count Lestocq and Prussian king Frederick the Great took an active part. The object was to strengthen the friendship between Prussia and Russia, to weaken the influence of Austria and to ruin the chancellor Aleksey Petrovich Bestuzhev-Ryumin, on whom Russian Empress Elizabeth relied, and who was a known partisan of the Austrian alliance. The diplomatic intrigue failed, largely due to the intervention of Sophie's mother, Johanna Elisabeth of Holstein-Gottorp. Historical accounts portray Johanna as a cold, abusive woman who loved gossip and court intrigues. Her hunger for fame centred on her daughter's prospects of becoming empress of Russia, but she infuriated Empress Elizabeth, who eventually banned her from the country for spying for King Frederick II of Prussia. Empress Elizabeth knew the family well: She had intended to marry Princess Johanna's brother Charles Augustus (Karl August von Holstein), but he died of smallpox in 1727 before the wedding could take place.'''
demo_question_beg = "when did he died?"
demo_answer = "in 1727"
demo_context = demo_context_beg
demo_question = demo_question_beg

demo_get_answer(demo_context, demo_question, demo_context_beg, demo_question_beg)
