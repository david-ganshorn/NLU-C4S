# -*- coding: utf-8 -*-

import tensorflow as tf

from question.handler import QuestionLoader, QuestionGenerator
from tagger.configuration import Variables
from tagger import model
from utils import constants, vocabulary, embeddings, dictionary as dict


def get_labels(sentence, session, tagger, configuration):
    config = configuration

    # Restore saved Models
    saver = tf.train.Saver()
    saver.restore(session, config.model_output)

    # Start an interactive Mode that let's you test your model
    idx_to_tag = {idx: tag for tag, idx in config.label_dict.str2idx.items()}

    # Process sentence for prediction
    low_sentence = sentence.lower()
    low_sentence = low_sentence.replace("?", "")
    low_words_raw = low_sentence.strip().split(" ")

    words = [config.word_dict.get_index(w) for w in low_words_raw]

    if type(words[0]) == tuple:
        words = zip(*words)

    pred_ids, _ = tagger.predict_batch(session, [words])
    preds = [idx_to_tag[idx] for idx in list(pred_ids[0])]

    return preds


def rest_api_call(question):

    config = Variables()
    constant = constants

    words = ['<UNK>', 'how', 'is', 'our', 'financial', 'performance', 'versus', 'analyst', 'estimates', 'which',
             'market', 'units', 'generate', 'the', 'most', 'cloud', 'revenue', 'have', 'had', 'highest', 'growth',
             'rates', 'region', 'generates', 'countries', 'generated', 'service', 'country', 'has', 'in', 'past',
             'half', 'year', 'unit', 'least', 'organization', 'last', 'quarter', 'location', 'for', 'locations',
             'greater', 'than', 'ten', 'million', 'show', 'all', 'that', 'profits', 'bigger', '1', 'are', 'top',
             'regarding', 'many', 'more', '10', 'percent', 'software', '5', 'on-premise', 'was', 'what', 'two',
             'quarters', 'during', 'us', 'development', 'of', 'throughout', 'three', 'margin', 'product', 'objects',
             'products', 'rolling', 'four', 'germany', 'rate', 'compared', 'to', 'oracle', 'tell', 'me', 'key',
             'driver', 'who', 'does', 'cost', 'structure', 'united', 'states', 'look', 'like', 'travel', 'and',
             'entertainment', 'expenses', 'much', 'license', 'over', 'hana', 'per', 'quarterly', 'services',
             'offerings', 'total', 'profit', 'first', 'north', 'america', 'this', 'years', 'trend', 'apj', 'evolved',
             'biggest', 'based', 'on', 'with', 'emea', 'share', 'decreasing', 'digits', 'compare', 'gross', 'between',
             'france', 'italy', 'categories', 'marketing', 'costs', 'differences', 'austria', 'calculate', 'each',
             'stake', '2016', 'display', 'high', 'percentage', 'difference', 'influencer', 'operating', 'dach',
             'lowest', 'earned', 'across', 'company', 'q1', 'worst', 'bottom', '3', 'relation', 'impact', 'looks',
             'traffic', 'same', 'indonesia', 'revenues', 'higher', 'figures', 'one', 'billion', 'dollars', 'great',
             'britain', 'china', 'comparison', 'regions', 'americas', 'double', 'digit', 'do', 'not', 'achieved',
             'rising', 'increasing', 'poland', 'number', 'a', 'lower', 'whole', 'sap', 'saudi', 'arabia',
             'profitability', 'japan', 'spend', 'croatia', 'research', 'denmark', 'latin', 'annually', 'hardware',
             'monthly', 'run', '4', 'spain', 'india', 'europe', 'strongest', 'split', 'by', 'sector', 'profitable',
             'business', 'margins', 'switzerland', 'belonging', 'south', 'numbers', 'runs', 'were', 'best',
             'performing', 'their', 'gains', '2017', 'distribution', 'africa', 'stagnating', 'month', '2',
             'organizations', 'having', 'positive', 'every', 's4', '2015', '2014', 'combined', 'q2', 'caused', 'erp',
             'usa', 'canada', 'q3', 'q4', 'five', 'deals', 'pipeline', 'facebook', 'dollar', 'sales', 'declining',
             'salesperson', 'closed', 'salespersons', 'salesmen', 'forecast', 'employee', 'sale', 'michael', 'scott',
             'employees', 'fulfilled', 'quotas', 'quota', 'goal', 'did', 'reach', 'brazil', 'current', 'quote', 'close',
             'ratio', 'target', 'opportunities', 'currently', 'bookings', 'average', 'deal', 'customers', 'we',
             'companies', 'consider', 'buy', 'boardroom', 'want', 'thinks', 'about', 'buying', 'missed', 'next',
             'volume', 'bought', 'new', 'won', 'as', 'well', 'ariba', 'concur', 'amount', 'salesman', 'sold',
             'licences', 'often', 'months', 'digital', 'r3', 'led', 'contracts', 'resulted', 'end', 'fourth', 'since',
             'licenses', 'within', 'selling', 'argentina', 'now', 'john', 'schneider', 'six', 'contract', 'volkswagen',
             'sealed', '80', 'values', 'resigned', 'from', '6', 'customer', 'loyalty', 'nestle', 'volumes', '250',
             'person', 'russia', 'leads', 'turned', 'into', 'sweden', 'items', 'opportunity', 'stage', 'booking',
             'weakest', 'related', 'holds', 'promises', 'promising', 'iot', 'mexico', 'asia', 'keeps', 'his', 'promise',
             'keep', 'analytics', 'analytic', 'conversion', 'responsible', 'thyssen', 'krupp', 'account', 'executive',
             'persons', 'lost', 'booked', 'starting', 'its', 'working', 'employed', 'open', 'or', 'representatives',
             'norway', 'name', 'representative', 'todd', 'packer', 'hr', 'successfactors', 'audi', 'australia',
             'porsche', 'peru', 'worldwide', 'seven', 'names', 'fifty', 'an', 'at&t', 'verizon', 'samsung', 'headcount',
             'people', 'israel', 'fte', 'palo', 'alto', 'department', 'developed', 'at', 'fully', 'loaded', 'plan',
             '2019', 'satisfaction', 'happiest', 'below', '95', 'dwight', 'schrute', 'managing', 'managed', 'wage',
             'managers', 'salary', 'position', 'level', 'newtown', 'square', 'early', 'talents', 'walldorf', 'learning',
             'completions', 'online', 'courses', 'completed', 'hired', 'fluctuation', 'thousand', 'women', 'management',
             'positions', 'leadership', 'manager', 'leader', 'men', 'managerial', 'responsibility', 'tasks', 'growing',
             'strong', 'fewest', 'external', 'workforce', 'consulting', 'non', 'billable', 'fastest', 'job', 'role',
             'diversity', 'gender', 'among', 'workers', 'hirings', 'so', 'far', 'joined', 'salaries', 'incomes',
             'students', 'bangalore', 'seattle', 'paris', '100', 'less', 'beijing', 'finance', 'administration',
             'innovation', 'human', 'resources', 'leaders', 'leading', 'frequent', 'hold', 'grouped', 'age', 'movement',
             'bill', 'mcdermott', 'greece', 'subordinates', 'finished', 'jimmy', 'kimmel', 'andy', 'bernard', 'where',
             'departments', 'available', 'developer', 'vacant', 'advertisement', 'earn', 'money', 'lot', 'christian',
             'klein', 'luka', 'mueller', 'head', 'been', 'signed', 'unlimited', 'conditions', 'mee', 'possibilities',
             'training', 'sessions', 'offered', 'nordics', 'singapore', 'none', '50000', 'older', '30', '50', 'l3',
             'threatening', 'competitors', 'competitor', 'threats', 'main', 'ibm', 'microsoft', 'also', 'facing',
             'gaining', 'momentum', 'against', 'netsuite', 'fiercest', 'workday', 'salesforce', 'in-memory', 'database',
             'investments', 'earnings', 'budget', 'list', 'dangerous', 'carries', 'invested', 'boosting', 'better',
             'stock', 'price', 'evaluation', 'index', 'exchange', 'google', 'january', 'december', 'may', 'right',
             'doing', 'prices', 'introduction', '8', 'value', 'capitalization', 'today', 'increased', 'saps', 'latest',
             'ibms', 'indices', 'apple', 'evaluate', 'mercedes', 'competing', 'compete', 'activities', 'activity',
             'year’s', 'r&d', 't&e', 'get', 'located', 'segment', 'minimum', 'maximum']
    labels = ['o', 'mea', 'cmp', 'res', 'argm', 'b_where', 'i_where', 'oper', 'grpby']

    # Build an empty dictionary for words and labels
    word_dict = dict.Dictionary()
    word_dict.initialize_unknown_token()

    label_dict = dict.Dictionary()

    for word in words:
        word_dict.add(word)

    config.word_dict = word_dict

    for label in labels:
        label_dict.add(label)

    config.label_dict = label_dict

    # Load word embeddings that map words in some language to high-dimensional vectors
    emb = embeddings.load(path=constant.data_folder, file=config.embedding_data, configuration=config)

    # Transpose embedding keys to corresponding keys in dictionary
    num_embeddings = embeddings.transpose(embeddings=emb, configuration=config)

    # Get only numeric vectors of embeddings
    vector_embeddings = embeddings.get_vectors(num_embeddings, configuration=config)

    # Build Tensorflow Graph
    tag_graph = tf.Graph()
    with tag_graph.as_default():
        # Build Seq2Tag Model
        # Create model and load parameters.
        tagger = model.SRLModel(config, vector_embeddings, config.label_dict.size())
        tagger.build()

    # Start Tensorflow Session
    tag_sess = tf.Session(graph=tag_graph)
    with tag_sess.as_default():
        with tag_graph.as_default():
            tf.global_variables_initializer().run()

            tags = get_labels(sentence=question, session=tag_sess, tagger=tagger, configuration=config)

    return tags
