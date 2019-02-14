"""

4. Given a string, find the longest substring without repeating characters along with the length.

     Input: "pwwkew"
     Output: wke,3

"""

chars = "pwwwkew"
len_chars = len(chars)

longest = 0
for i in range(len_chars):
    seen = set()
    for j in range(i, len_chars):
        if chars[j] in seen:
            all_words = ''.join(str(s) for s in seen)
            break
        seen.add(chars[j])

    # print(all_words, seen)
    longest = max(len(seen), longest)

first_1 = 0

for i in range(len_chars):
    seen = set()
    for j in range(i, len_chars):
        if chars[j] in seen:
            if len(seen) == longest:
                first_1 = j
                break
            break
        seen.add(chars[j])

# print(all_words , ',', longest)
print(first_1, longest)
ksee = ""
for i in range(first_1-longest, first_1):
    ksee = ksee + chars[i]
print('Output: ', ksee, ',', longest)


