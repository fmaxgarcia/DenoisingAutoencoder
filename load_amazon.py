import os


def parse_reviews(filename, score):
    texts, scores = [], []

    element = ""
    new_review = False
    text = ""
    reviews = open(filename, "r").readlines()
    for i, line in enumerate(reviews):
        line = line[:-1]
        if line == '<review_text>':
            new_review = True
        elif line == '</review_text>':
            try:
                text.encode('utf-8')
                texts.append(text)
                scores.append( score )
            except:
                print "Encoding error: Ignoring review"
            new_review = False
            text = ""
        elif new_review == True:
            text += line

    return texts, scores

def load_data(dataset_path, domain):
    corpus, ratings = [], []
    
    negative_filename = dataset_path+domain+"/negative.review"
    positive_filename = dataset_path+domain+"/positive.review"

    neg_texts, neg_ratings = parse_reviews(negative_filename, score=0)
    pos_texts, pos_ratings = parse_reviews(positive_filename, score=1)

    corpus.extend(neg_texts)
    ratings.extend(neg_ratings)
    
    corpus.extend(pos_texts)
    ratings.extend(pos_ratings)
    return corpus, ratings


        