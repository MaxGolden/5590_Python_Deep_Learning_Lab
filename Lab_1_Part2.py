
student_list = [( "John", ("Physics", 80)) , ("Daniel", ("Science", 90)), ("John",("Science", 95)), ("Mark",("Maths", 100)), ("Daniel", ("History", 75)), ("Mark", ("Social", 95))]
student_dict = {}

for student in student_list:
    if student[0] not in student_dict:
        student_dict[student[0]] = list()
        student_dict[student[0]].append(student[1])
    else:
        student_dict[student[0]].append(student[1])

for student in student_dict :
    print (student,":", student_dict[student])