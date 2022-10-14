from editdistance import distance


def calc_cer(target_text, predicted_text) -> float:
    if not len(target_text):
        return 1
    return distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_text = target_text.split(' ')
    if not len(target_text):
        return 1
    return distance(target_text, predicted_text.split(' ')) / len(target_text)
