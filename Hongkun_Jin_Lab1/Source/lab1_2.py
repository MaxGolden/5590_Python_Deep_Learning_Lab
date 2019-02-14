"""
lab1 - 2
Suppose you have a list of tuples as follows:

      [( ‘John’, (‘Physics’, 80)) , (‘ Daniel’, (‘Science’, 90)), (‘John’, (‘Science’, 95)),
      (‘Mark’,(‘Maths’, 100)), (‘Daniel’, (’History’, 75)), (‘Mark’, (‘Social’, 95))]

    Create a dictionary with keys as names and values as list of (subjects, marks) in sorted order.

       {
            John : [(‘Physics’, 80), (‘Science’, 95)]
            Daniel : [ (’History’, 75), (‘Science’, 90)]
            Mark : [ (‘Maths’, 100), (‘Social’, 95)]
       }
"""
tuple1 = [('John', ('Physics', 80)), ('Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark', ('Maths', 100)),
          ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]

# for x in tuple1:
#     print(x)
# tuple1[1] = 5
# print(tuple1)
# print(tuple1[1])
# Python code to convert into dictionary


def convert_set(tup, dic):
    for a, b in tup:
        dic.setdefault(a, []).append(b)
    return dic


# Driver Code
dictionary = {}
tuple_dic = convert_set(tuple1, dictionary)
print(tuple_dic)


