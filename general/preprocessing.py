import string
import re
import emoji

def preprocessing(post):
    punctuation = string.punctuation
    
    #remove unicode values
    post = post.encode("ascii", "ignore")
    post = post.decode()
    
    #remove emojis
    post = emoji.get_emoji_regexp().sub(r'', post)
    
    #remove urls
    post = [word for word in post.split() if 'http' not in word.lower() and 'www' not in word.lower()]

    #remove mentions and hashtags based on their positions in the sentence,
    #clusters of hashtags and/or mentions are removed
    first = False
    for i in range(len(post),-1,-1):
        if len(post) < 2:
            return None
        try:
            if i == 0:
                if '#' in post[0] or '@gebruiker' in post[0]:
                    if first:
                        del post[i]
                    elif post[1] != post[1].lower():
                        del post[i]
            elif i == 1:
                if '#' in post[i] or '@gebruiker' in post[i]:
                    if '#' in post[i - 1] or '.' in post[i - 1] or '?' in post[i - 1] or '!' in post[i - 1] or '@gebruiker' in post[i - 1]:
                        del post[i]
                        first = True
            elif i == len(post):
                if '#' in post[-1] or '@gebruiker' in post[-1]:
                    if '#' in post[-2] or '.' in post[-2] or '!' in post[-2] or '?' in post[-2] or '@gebruiker' in post[-2]:
                        del post[-1]
            else:
                if '#' in post[i] or '@gebruiker' in post[i]:
                    if '#' in post[i - 1] or '.' in post[i - 1] or '?' in post[i - 1] or '!' in post[i - 1] or '@gebruiker' in post[i - 1]:
                        del post[i]
        except IndexError:
            break

    post = ' '.join(post)
    
    #add spacing around punctuation
    post = ''.join([' ' + letter + ' ' if letter in punctuation else letter for letter in post])
    
    #remove redundant spaces
    post = re.sub(' +', ' ', post)
    if post[0] == ' ':
        post = post[1:]
    if len(post.split()) < 2:
        return None
#     if len(post.split()) > 32:
#         return None
    return post