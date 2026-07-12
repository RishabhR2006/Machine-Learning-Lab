def get_highest_char(text_str):
    letters=[char.lower() for char in text_str if char.isalpha()]
    if not letters:
        return "No alphabetic characters found.",0
    counts={}
    for char in letters:
        counts[char]=counts.get(char,0)+1
    top_char=max(counts,key=counts.get)
    top_count=counts[top_char]
    return top_char,top_count

word_input="hippopotamus"
frequent_char,occurrence=get_highest_char(word_input)
print(f"Input: '{word_input}'")
print(f"Max char is '{frequent_char}' & count is {occurrence}.")