import json
import csv

#Import this function
def clean_json_and_save_csv(streamer_name, date, path_prefix='/content/drive/MyDrive/twitch_chat/'):
    """Primary cleaning function, takes in json created by lay (github.com/lay295/TwitchDownloader)
        writes cleaned csvs for the comments and emotes of the stream.
        
        Assumes jsons are saved as .../<streamer_name>/<streamer_name><month-day>.json
        So for streamer Yeatle on August first, the json should be saved as '.../Yeatle/Yeatle8-1.json
            (date does not need to be a date it can be anything as long as you're consistent)

        The csvs are saved in the same directory.
    """

    with open(path_prefix + streamer_name +'/'+ streamer_name + date + '.json') as f:
        stream_json = json.load(f)

    #get the lists of comments and emotes from the json
    cleaned_comments, cleaned_emotes = clean_comments(stream_json)

    #write the comments and emotes to csv files

    with open(path_prefix + streamer_name +'/'+ streamer_name + date + 'comments.csv', "w", newline="") as f:
        writer = csv.writer(f)
        #header
        writer.writerow(['time', 'comment_text'])
        #write rows
        writer.writerows(cleaned_comments)

    with open(path_prefix + streamer_name +'/'+ streamer_name + date + 'emotes.csv', "w", newline="") as f:
        writer = csv.writer(f)
        #header
        writer.writerow(['time', '_id', 'emote_text'])
        #write rows
        writer.writerows(cleaned_emotes)


def clean_comment(comment):
    #gather relevant info
    comment_time = comment['content_offset_seconds']
    comment_text = comment['message']['body']
    comment_emotes_list = comment['message']['emoticons']
    comment_emotes = {}
    #parse the unique emotes used in the comment
    if comment_emotes_list is not None:
        for emote_info in comment_emotes_list:
            _id, begin, end = emote_info['_id'], emote_info['begin'], emote_info['end']
            if _id not in comment_emotes.keys():
                comment_emotes[_id] = comment_text[begin:end+1]
    return comment_time, comment_text, comment_emotes

def filter_comment(comment):
    
    #gathering relevant info.
    name = comment['commenter']['name'].lower()
    text = comment['message']['body'].lower()
    msg_id = comment['message']['user_notice_params']['msg-id']

    #is comment a subscriber comment?
    if msg_id and 'sub' in msg_id.lower():
        return True
    #is commenter a twitch bot?
    if 'bot' in name:
        return True
    #similar, targets 'StreamElements' bot
    elif 'elements' in name:
        return True
    #filters out replies
    #TODO though it does also filter out messages that @ the streamer, maybe want to keep those.
    elif '@' in text:
        return True
    #Otherwise let the comment through
    else:
        False
    

def clean_comments(stream_json):
    
    cleaned_comments = []
    cleaned_emotes = []
    comments = stream_json['comments']
    for comment in comments:
        #check if comment should be recorded
        if filter_comment(comment):
            continue
        #gets comment time, text, and emotes (repeated emotes are ignored)
        comment_time, comment_text, comment_emotes = clean_comment(comment)
        #add comment row
        cleaned_comments.append([comment_time, comment_text])
        #turn used emotes from dict to list and add rows
        for _id, emote_text in comment_emotes.items():
            cleaned_emotes.append([comment_time, _id, emote_text])

    return cleaned_comments, cleaned_emotes

