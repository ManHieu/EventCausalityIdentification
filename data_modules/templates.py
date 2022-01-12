TEMPLATES = {
    'eci': [   
    # context is sentences, hypothesis has form "event causes event", option is yes, no.
    # Answer is yes or no and dependency path 
    ("{context}\n\nBased on the paragraph above can we conclude that \"There is a causal relationship between the events {head} and {tail}\"?", "{answer}"),
    ("{context}\n\nCan we infer the following?\nThere is a causal relationship between the events {head} and {tail}", "{answer}"),
    # conclusion is event cause event or None
    # ("Generate a conclusion from the context: \n Context: {context}", "{conclusion}"),
    # Question has form: Does event cause event
    ("Answer based on context:\n\n{context}\n\nIs there a causal relationship between the events {head} and {tail}?", "{answer}"),
    ("{context}\n\nIs there a causal relationship between the events {head} and {tail}?", "{answer}"),
    ("{context}\n\nIs it true that there is a causal relationship between the events {head} and {tail}?", "{answer}"),
    ("{context}\n\nIs there a causal relation between {head} and {tail}?", "{answer}"), # *
    # context has form: sentence + event pair is event and event
    ("{context}\n\nDoes causality relation exist between the events {head} and {tail}?", "{answer}"),
    # baseline
    ("{context}\n\n{head} and {tail}?", "{answer}"),]
}