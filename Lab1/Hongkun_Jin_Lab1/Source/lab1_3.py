"""
3. Consider the following scenario. You have a list of students who are attending class "Python" and another
list of students who are attending class "Web Application".

•	Find the list of students who are attending both the classes.
•	 Also find the list of students who are not common in both the classes.

     Print the both lists. Consider accepting the input from the console for list of students that belong to class
“Python” and class “Web Application”.

"""

class_python = ["Max", "brand", "Jack", "Lily", "Gigi"]
class_webApp = ["Max", "Kailey", "Gigi", "Fill", "Amy"]

# both_class = set(class_python) - (set(class_python) - set(class_webapp))
both_class = set(class_python) & set(class_webApp)
diff_class = set(class_python) ^ set(class_webApp)
# diff_class = (set(class_python) - set(class_webapp)) | ((set(class_webapp) - set(class_python))
print('Students who are attending both the classes \n', both_class, '\n')
print('Students who are not common in both the classes \n', diff_class)
