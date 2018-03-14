
import tensorflow as tf

from question.handler import QuestionLoader, QuestionGenerator
from tagger.configuration import Variables
from tagger import model
from utils import constants, vocabulary, embeddings


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

    # Load questions from disk
    loader = QuestionLoader(path=constant.data_folder, file=constants.training_data)
    data = loader.load_data()

    # PreProcess the questions and generate features
    generator = QuestionGenerator(data=data, configuration=config)
    questions = generator.pre_processing()

    # Construct Vocabulary
    x = vocabulary.build_words(questions=questions, configuration=config)
    y = vocabulary.build_labels(questions=questions, configuration=config)

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
